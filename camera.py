import numpy

__all__ = (
        'Camera',
        'get_camera',
)

class Camera(object):
    """
    Representation of a camera's intrinsic parameters.

    """
    def __init__(self, fc, cc, alpha_c, kc, size):
        # See http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
        # for explanations of these parameters.
        self._KK = numpy.matrix([[fc[0], alpha_c * fc[0], cc[0]],
                                 [0.0,   fc[1],           cc[1]],
                                 [0.0,   0.0,             0.0]])
        self.size = size
    
    def camera_pos_to_pixel(self, camera_pos, apply_distortion=False):
        """
        Convert a camera space vector into a pixel coordinate.
        
        Distortion is not yet supported.

        :param camera_pos:
            3x1 vector representing the camera position in vector space.
        
        :param apply_distortion:
            If False, distortion parameters are assumed to be zero.

        :returns:
            A 2x1 vector representing the pixel coordinates.
        
        """
        if apply_distortion:
            raise NotImplementedError

        projected = camera_pos / camera_pos[2, 0]
        pixel = self._KK * projected

        return pixel[:2,:]

_db = {
    'nexus_4_photo': Camera(fc=[2868., 2872.,],
                            cc=[1219., 1591.,],
                            alpha_c=0.0,
                            kc=[0.18, -0.768, 0.00145, 0.00030, 0.0],
                            size=(2448, 3264)),
    'nexus_4_video': Camera(fc=[(1280. / 3264) * 2868., (1280. / 3264) * 2872.,],
                            cc=[(1280. / 3264) * 1219. - 120, (1280. / 3264) * 1591.,],
                            alpha_c=0.0,
                            kc=[0.18, -0.768, 0.00145, 0.00030, 0.0], #kc may need scaling?
                            size=(720, 1280))
    }


def get_camera(camera_name):
    """Obtain a camera's intrinsic parameters by name."""
    return _db[camera_name]
