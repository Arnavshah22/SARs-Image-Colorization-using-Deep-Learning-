from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import io
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import uuid
import os
import shutil

# === FastAPI App Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Flattened ConvBlock Model Matching .pth File ===

class DownsamplingBlock(nn.Module):
    """Defines the Unet downsampling block. 
    
    Consists of Convolution-BatchNorm-ReLU layer with k filters.
    """
    def __init__(self, c_in, c_out, kernel_size=4, stride=2, 
                 padding=1, negative_slope=0.2, use_norm=True):
        """
        Initializes the UnetDownsamplingBlock.
        
        Args:
            c_in (int): The number of input channels.
            c_out (int): The number of output channels.
            kernel_size (int, optional): The size of the convolving kernel. Default is 4.
            stride (int, optional): Stride of the convolution. Default is 2.
            padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
            negative_slope (float, optional): Negative slope for the LeakyReLU activation function. Default is 0.2.
            use_norm (bool, optinal): If use norm layer. If True add a BatchNorm layer after Conv. Default is True.
        """
        super(DownsamplingBlock, self).__init__()
        block = []
        block += [nn.Conv2d(in_channels=c_in, out_channels=c_out,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=(not use_norm) # No need to use a bias if there is a batchnorm layer after conv
                          )]
        if use_norm:
            block += [nn.BatchNorm2d(num_features=c_out)]
        
        block += [nn.LeakyReLU(negative_slope=negative_slope)]

        self.conv_block = nn.Sequential(*block)
        
    def forward(self, x):
        return self.conv_block(x)

class UpsamplingBlock(nn.Module):
    """Defines the Unet upsampling block.
    """
    def __init__(self, c_in, c_out, kernel_size=4, stride=2, 
                 padding=1, use_dropout=False, use_upsampling=False, mode='nearest'):
        
        """
        Initializes the Unet Upsampling Block.
        
        Args:
            c_in (int): The number of input channels.
            c_out (int): The number of output channels.
            kernel_size (int, optional): Size of the convolving kernel. Default is 4.
            stride (int, optional): Stride of the convolution. Default is 2.
            padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
            use_dropout (bool, optional): if use dropout layers. Default is False.
            upsample (bool, optinal): if use upsampling rather than transpose convolution. Default is False.
            mode (str, optional): the upsampling algorithm: one of 'nearest', 
                'bilinear', 'bicubic'. Default: 'nearest'
        """
        super(UpsamplingBlock, self).__init__()
        block = []
        if use_upsampling:
            # Transpose convolution causes checkerboard artifacts. Upsampling
            # followed by a regular convolutions produces better results appearantly
            # Please check for further reading: https://distill.pub/2016/deconv-checkerboard/
            # Odena, et al., "Deconvolution and Checkerboard Artifacts", Distill, 2016. http://doi.org/10.23915/distill.00003
            
            mode = mode if mode in ('nearest', 'bilinear', 'bicubic') else 'nearest'
            
            block += [nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(in_channels=c_in, out_channels=c_out,
                          kernel_size=3, stride=1, padding=padding,
                          bias=False
                          )
                )]
        else:
            block += [nn.ConvTranspose2d(in_channels=c_in, 
                                         out_channels=c_out,
                                         kernel_size=kernel_size, 
                                         stride=stride,
                                         padding=padding, bias=False
                                         )
                     ]
        
        block += [nn.BatchNorm2d(num_features=c_out)]

        if use_dropout:
            block += [nn.Dropout(0.5)]
            
        block += [nn.ReLU()]

        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)
    
class UnetDecoder(nn.Module):
    """Creates the Unet Decoder Network.
    """
    def __init__(self, c_in=512, c_out=64, use_upsampling=False, mode='nearest'):
        """
        Constructs the Unet Decoder Network.

        Ck denote a Convolution-BatchNorm-ReLU layer with k filters.
        
        CDk denotes a Convolution-BatchNorm-Dropout-ReLU layer with a dropout rate of 50%.
            CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        Args:
            c_in (int): Number of input channels.
            c_out (int, optional): Number of output channels. Default is 512.
            use_upsampling (bool, optional): Upsampling method for decoder. 
                If True, use upsampling layer followed regular convolution layer.
                If False, use transpose convolution. Default is False
            mode (str, optional): the upsampling algorithm: one of 'nearest', 
                'bilinear', 'bicubic'. Default: 'nearest'
        """
        super(UnetDecoder, self).__init__()
        self.dec1 = UpsamplingBlock(c_in, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode) # CD512
        self.dec2 = UpsamplingBlock(1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode) # CD1024
        self.dec3 = UpsamplingBlock(1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode) # CD1024
        self.dec4 = UpsamplingBlock(1024, 512, use_upsampling=use_upsampling, mode=mode) # C1024
        self.dec5 = UpsamplingBlock(1024, 256, use_upsampling=use_upsampling, mode=mode) # C1024
        self.dec6 = UpsamplingBlock(512, 128, use_upsampling=use_upsampling, mode=mode) # C512
        self.dec7 = UpsamplingBlock(256, 64, use_upsampling=use_upsampling, mode=mode) # C256
        self.dec8 = UpsamplingBlock(128, c_out, use_upsampling=use_upsampling, mode=mode) # C128
    

    def forward(self, x):
        x9 = torch.cat([x[1], self.dec1(x[0])], 1) # (N,1024,H,W)
        x10 = torch.cat([x[2], self.dec2(x9)], 1) # (N,1024,H,W)
        x11 = torch.cat([x[3], self.dec3(x10)], 1) # (N,1024,H,W)
        x12 = torch.cat([x[4], self.dec4(x11)], 1) # (N,1024,H,W)
        x13 = torch.cat([x[5], self.dec5(x12)], 1) # (N,512,H,W)
        x14 = torch.cat([x[6], self.dec6(x13)], 1) # (N,256,H,W)
        x15 = torch.cat([x[7], self.dec7(x14)], 1) # (N,128,H,W)
        out = self.dec8(x15) # (N,64,H,W)
        return out
    
class UnetEncoder(nn.Module):
    """Create the Unet Encoder Network.
    
    C64-C128-C256-C512-C512-C512-C512-C512
    """
    def __init__(self, c_in=3, c_out=512):
        """
        Constructs the Unet Encoder Network.

        Ck denote a Convolution-BatchNorm-ReLU layer with k filters.
            C64-C128-C256-C512-C512-C512-C512-C512
        Args:
            c_in (int, optional): Number of input channels.
            c_out (int, optional): Number of output channels. Default is 512.
        """
        super(UnetEncoder, self).__init__()
        self.enc1 = DownsamplingBlock(c_in, 64, use_norm=False) # C64
        self.enc2 = DownsamplingBlock(64, 128) # C128
        self.enc3 = DownsamplingBlock(128, 256) # C256
        self.enc4 = DownsamplingBlock(256, 512) # C512
        self.enc5 = DownsamplingBlock(512, 512) # C512
        self.enc6 = DownsamplingBlock(512, 512) # C512
        self.enc7 = DownsamplingBlock(512, 512) # C512
        self.enc8 = DownsamplingBlock(512, c_out) # C512

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        out = [x8, x7, x6, x5, x4, x3, x2, x1] # latest activation is the first element
        return out
    

class Generator(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, c_in=3, c_out=3, use_upsampling=False, mode='nearest'):
        """
        Constructs a Unet generator
        Args:
            c_in (int): The number of input channels.
            c_out (int): The number of output channels.
            use_upsampling (bool, optional): Upsampling method for decoder. 
                If True, use upsampling layer followed regular convolution layer.
                If False, use transpose convolution. Default is False
            mode (str, optional): the upsampling algorithm: one of 'nearest', 
                'bilinear', 'bicubic'. Default: 'nearest'
        """
        super(Generator, self).__init__()
        self.encoder = UnetEncoder(c_in=c_in)
        self.decoder = UnetDecoder(use_upsampling=use_upsampling, mode=mode)
        # In the paper, the authors state:
        #   """
        #       After the last layer in the decoder, a convolution is applied
        #       to map to the number of output channels (3 in general, except
        #       in colorization, where it is 2), followed by a Tanh function.
        #   """
        # However, in the official Lua implementation, only a Tanh layer is applied.
        # Therefore, I took the liberty of adding a convolutional layer with a 
        # kernel size of 3.
        # For more information please check the paper and official github repo:
        # https://github.com/phillipi/pix2pix
        # https://arxiv.org/abs/1611.07004
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=c_out,
                      kernel_size=3, stride=1, padding=1,
                      bias=True
                      ), 
            nn.Tanh()
            )
    
    def forward(self, x):
        outE = self.encoder(x)
        outD = self.decoder(outE)
        out = self.head(outD)
        return out

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
model_path = os.path.join('pix2pix_gen_265.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
# === Inference Helper ===
def process_image(image_path: str) -> Image.Image:
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).expand(-1, 3, -1, -1).to(device)

    with torch.no_grad():
        output = model(tensor)[0].cpu()
        output = (output + 1) / 2
    return transforms.ToPILImage()(output)

# === Upload Endpoint ===
@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Save uploaded image
        input_path = "uploaded_image.png"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Process with model
        output_image = process_image(input_path)
        output_path = "output_image.png"
        output_image.save(output_path)

        # Return output image
        return FileResponse(output_path, media_type="image/png", filename="colorized_output.png")

    except Exception as e:
        print("BACKEND ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

        raise HTTPException(status_code=500, detail=str(e))