#!/usr/bin/env python

import cv2
import sys
import math
import numpy
import argparse
import numpy.linalg

import camera

# Parse arguments
def float_parser(string):
    while string[0] == '\\':
        string = string[1:]
    return float(string)

def make_tuple_parser(el_parser, sep=','):
    def parser(string):
        return tuple(el_parser(x.strip()) for x in string.split(sep) if x.strip())
    return parser

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-video', help="Input video", required=True)
parser.add_argument('-o', '--output-format', help="Output format", default="out%06d.png")
parser.add_argument('-c', '--camera-name', help="Camera name", required=True)
parser.add_argument('-r', '--rotate', help="Rotate input 90 degrees CCW n times", type=int, default=0)
parser.add_argument('-s', '--output-size', help="Size in pixels of output", type=make_tuple_parser(int, sep='x'), default=(500,1000))
parser.add_argument('-p', '--pixel-size', help="Metres per output pixel", type=float, default=0.02)
parser.add_argument('-H', '--camera-height', help="Camera height in metres", type=float, default=1.0)
parser.add_argument('-g', '--gravity-vector', help="Gravity vector", type=make_tuple_parser(float_parser), required=True)
parser.add_argument('--start-frame', help="Start frame", type=int, default=0)
parser.add_argument('--end-frame', help="End frame", type=int, default=-1)
args = parser.parse_args()

cam = camera.get_camera(args.camera_name)

# `camera_grav` is the direction of gravity in the camera's coordinate system.
camera_grav = numpy.matrix([list(args.gravity_vector)]).T
camera_grav[2,0] = -camera_grav[2,0]   # We want +Z to be in the direction of the camera
camera_grav = camera_grav / numpy.linalg.norm(camera_grav)

# `C` is the world-space to camera-space matrix.
# Choose `C` such that C * (0,-1,0).T = camera_grav, and C * (0,0,1).T is in the Y,Z plane.
C = numpy.matrix(numpy.zeros((3, 3)))
C[:,1] = -camera_grav
C[:,0] = numpy.matrix([[1.], [-C[0,1] / C[1,1]], [0]])
C[:,0] = C[:,0] / numpy.linalg.norm(C[:,0])
C[:,2] = numpy.matrix(numpy.cross(C[:,0].T, C[:,1].T).T)

# Make map_x and map_y to map output pixels to input pixels.
print "Building maps"
map_x = numpy.zeros(args.output_size).T
map_y = numpy.zeros(args.output_size).T
for y in range(args.output_size[1]):
    for x in range(args.output_size[0]):
        road_world = numpy.matrix([[args.pixel_size * (float(x) - args.output_size[0]/2.)],
                                   [-args.camera_height],
                                   [args.pixel_size * float(y)]])

        road_pixel = cam.camera_pos_to_pixel(C * road_world)

        map_y[args.output_size[1] - y - 1, x] = cam.size[1] - road_pixel[1, 0]
        map_x[args.output_size[1] - y - 1, x] = road_pixel[0, 0]
print "Done"

# Convert the input images.
vc = cv2.VideoCapture()
vc.open(args.input_video)

def frame_num_gen():
    i = args.start_frame
    while True:
        if args.end_frame >= 0 and i >= args.end_frame:
            break
        yield i

        i += 1

for i in frame_num_gen():
    if not vc.grab():
        break
       
    print "Processing {}".format(i)
    im_in = vc.retrieve()[1]
    im_in = numpy.rot90(im_in, args.rotate)
    assert list(im_in.shape) == list(reversed(cam.size)) + [3], "{} != {}".format(im_in.shape, cam.size)
    im_out = cv2.remap(im_in, map_x.astype('float32'), map_y.astype('float32'), cv2.INTER_CUBIC)
    cv2.imwrite(args.output_format % i, im_out)

