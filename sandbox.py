from tkinter import filedialog
from math import ceil
import math
from matplotlib.image import imread, imsave
from jinja2 import Template
import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule
from tkinter import *
from tkinter import *
from PIL import Image, ImageTk

chosen_file = '/home/data/Pictures/fox-small.jpg'
chosen_effect = "Gamma 2.0"

def test_gamma_half(img):
    return(cpu_gamma_half(img), gpu_gamma_half(img))

def gamma_half(px):
    px = px / 255;
    corrected = math.pow(px, 0.5)
    if px > 1.0:
        px = 1.0
    if px < 0.0:
        px = 0.0
    corrected *= 255.0
    return round(corrected)

def cpu_gamma_half(img):
    fn = numpy.vectorize(gamma_half)
    return fn(img)

def gpu_gamma_half(img):
    template = Template("""
    __global__ void gamma_two(unsigned char *dest, unsigned char *img) {
        const int row = threadIdx.x + blockDim.x*blockIdx.x;
        const int col = threadIdx.y + blockDim.y*blockIdx.y;
        const int chan = threadIdx.z;

        const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};

        if (idx > {{width}}*{{height}}*{{depth}}) {
            return;
        }

        const float px = img[idx] / 255.0f;
        float out = __powf(px, 0.5f);

        if (out > 1.0f){
            out = 1.0f;
        }
        if (out < 0.0f) {
            out = 0.0f;
        }

        dest[idx] = __float2int_rn(out*255.0f);
    }
    """)

    width, height, depth = img.shape
    module = SourceModule(template.render(width=width, height=height, depth=depth))

    dest = numpy.zeros_like(img)

    gpu_brighten = module.get_function("gamma_two")
    gpu_brighten(drv.Out(dest), drv.In(img), block=(8,8,3), grid=(ceil(width/8), ceil(height/8)))

    return dest



def test_gamma_two(img):
    return(cpu_gamma_two(img), gpu_gamma_two(img))

def gamma_two(px):
    px = px / 255;
    corrected = math.pow(px, 2.0)
    if px > 1.0:
        px = 1.0
    if px < 0.0:
        px = 0.0
    corrected *= 255.0
    return round(corrected)

def cpu_gamma_two(img):
    fn = numpy.vectorize(gamma_two)
    return fn(img)

def gpu_gamma_two(img):
    template = Template("""
    __global__ void gamma_two(unsigned char *dest, unsigned char *img) {
        const int row = threadIdx.x + blockDim.x*blockIdx.x;
        const int col = threadIdx.y + blockDim.y*blockIdx.y;
        const int chan = threadIdx.z;

        const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};

        if (idx > {{width}}*{{height}}*{{depth}}) {
            return;
        }

        const float px = img[idx] / 255.0f;
        float out = __powf(px, 2);

        if (out > 1.0f){
            out = 1.0f;
        }
        if (out < 0.0f) {
            out = 0.0f;
        }

        dest[idx] = __float2int_rn(out*255.0f);
    }
    """)

    width, height, depth = img.shape
    module = SourceModule(template.render(width=width, height=height, depth=depth))

    dest = numpy.zeros_like(img)

    gpu_brighten = module.get_function("gamma_two")
    gpu_brighten(drv.Out(dest), drv.In(img), block=(8,8,3), grid=(ceil(width/8), ceil(height/8)))

    return dest

def test_brighten(img):
    return (cpu_brighten(img), gpu_brighten(img))

def cpu_brighten(img):
    fn = numpy.vectorize(brighten_vectorized)
    return fn(img)


def gpu_brighten(img):
    template = Template("""
    __global__ void brighten(unsigned char *dest, unsigned char *img)
    {
        const int row = threadIdx.x + blockDim.x*blockIdx.x;
        const int col = threadIdx.y + blockDim.y*blockIdx.y;
        const int chan = threadIdx.z;

        const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};
        if (idx > {{width}}*{{height}}*{{depth}}) {
            return;
        }
        const int px = img[idx];
        const int temp = px*2;

        if (0 > temp){
            dest[idx] = 0;
            return;
        }
        if (255 < temp){
            dest[idx] = 255;
            return;
        }
        dest[idx] = temp;
    }
    """)

    width, height, depth = img.shape
    module = SourceModule(template.render(width=width, height=height, depth=depth))

    dest = numpy.zeros_like(img)

    gpu_brighten = module.get_function("brighten")
    gpu_brighten(drv.Out(dest), drv.In(img), block=(8,8,3), grid=(ceil(width/8), ceil(height/8)))

    return dest


def brighten_vectorized(px):
    temp = px * 2
    return clamp(0, 255, temp)

def edge_detect(img):
    result = numpy.zeros_like(img)
    width, height, depth = img.shape
    for row in range(1, width-1):
        for col in range(1, height-1):
            total = 0
            for chan in range(depth):
                total += 8*img[row,col,chan]

                total -= img[row-1,col-1,chan]
                total -= img[row-1,col,chan]
                total -= img[row-1,col+1,chan]

                total -= img[row,col-1,chan]
                total -= img[row,col+1,chan]

                total -= img[row+1,col-1,chan]
                total -= img[row+1,col,chan]
                total -= img[row+1,col+1,chan]

            result[row,col,0] = clamp(0,255,total)
            result[row,col,1] = clamp(0,255,total)
            result[row,col,2] = clamp(0,255,total)

    return result

def gpu_edge_detect(img):
    template = Template("""
    __global__ void edge(unsigned char *dest, unsigned char *img)
    {
        const int row = threadIdx.x + blockDim.x*blockIdx.x;
        const int col = threadIdx.y + blockDim.y*blockIdx.y;

        if (row < 1 || row >= {{width}} - 1 ||
            col < 1 || col >= {{height}} - 1) {
            return;
        }

        int total = 0;
        for (int chan = 0; chan < {{depth}}; chan++) {
            const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};

            const int down = {{depth}}*{{height}};
            const int right = {{depth}};

            total += 8 * img[idx];

            total -= img[idx - down - right];
            total -= img[idx - down];
            total -= img[idx - down + right];

            total -= img[idx - down];
            total -= img[idx + down];

            total -= img[idx + down - right];
            total -= img[idx + down];
            total -= img[idx + down + right];

        }

        const int chan = threadIdx.z;
        const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};

        if (0 > total){
            dest[idx] = 0;
            dest[idx+1] = 0;
            dest[idx+2] = 0;
            return;
        }
        if (255 < total){
            dest[idx] = 255;
            dest[idx+1] = 255;
            dest[idx+2] = 255;
            return;
        }
        dest[idx] = total;
        dest[idx+1] = total;
        dest[idx+2] = total;
    }
    """)


    width, height, depth = img.shape
    module = SourceModule(template.render(width=width, height=height, depth=depth))

    dest = numpy.zeros_like(img)
    gpu_brighten = module.get_function("edge")
    gpu_brighten(drv.Out(dest), drv.In(img), block=(8,8,1), grid=(ceil(width/8), ceil(height/8)))

    return dest


def clamp(low, high, val):
    if low > val:
        return low
    if high < val:
        return high
    return val

def test_edge_detection(img):
    return (edge_detect(img), gpu_edge_detect(img))


switch = {
    'Edge Detection': test_edge_detection,
    'Brighten x2': test_brighten,
    'Gamma 2.0': test_gamma_two,
    'Gamma 0.5': test_gamma_half,
}


img = imread(chosen_file)

print(img.shape)
cpu_render, gpu_render = switch[chosen_effect](img)
imsave('/tmp/cpu_render.png', cpu_render)
imsave('/tmp/gpu_render.png', gpu_render)



class ImgFrame(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)
        self.original = Image.open('/tmp/gpu_render.png')
        self.image = ImageTk.PhotoImage(self.original)
        self.display = Canvas(self, bd=0, highlightthickness=0)
        self.display.create_image(0, 0, image=self.image, anchor=NW, tags="IMG")
        self.display.grid(row=0, sticky=W+E+N+S)
        self.pack(fill=BOTH, expand=1)
        self.bind("<Configure>", self.resize)

    def resize(self, event):
        w, h = self.original.size
        ew, eh = event.width, event.height

        ratio = w/h

        if ew < eh * ratio:
            size = (round(eh*ratio), eh)
        else:
            size = (ew, round(ew/ratio))

        resized = self.original.resize(size,Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(resized)
        self.display.delete("IMG")
        self.display.create_image(0, 0, image=self.image, anchor=NW, tags="IMG")

root = Tk()
app = ImgFrame(root)
app.mainloop()