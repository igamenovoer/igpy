# blend images
from enum import Enum
import numpy as np
from attrs import define, field


class BlendMode(Enum):
    NORMAL = 0  # alpha blending
    SCREEN = 1  # screen blending
    OVERLAY = 2  # overlay blending
    MULTIPLY = 3  # multiply blending
    COLOR_BURN = 4  # color burn blending
    LINEAR_BURN = 5  # linear burn blending
    DIVIDE = 6  # divide blending
    SOFT_LIGHT = 7  # soft light blending
    HARD_LIGHT = 8  # hard light blending


@define(eq=False)
class ImageBlender:
    """
    Blend two images using the specified blend mode and alpha value,
    the foreground image is overlayed on top of the background image, then the alpha is applied
    to the foreground image, and then the two images are blended using the specified blend mode
    """

    foreground_image: np.ndarray = field(
        converter=lambda x: ImageBlender.to_rgba(np.array(x))
    )
    background_image: np.ndarray = field(
        converter=lambda x: ImageBlender.to_rgba(np.array(x))
    )
    blend_mode: BlendMode = field(default=BlendMode.NORMAL)

    # alpha of the foreground image
    alpha: float = field(default=1.0)

    @staticmethod
    def to_rgba(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        if image.ndim == 2:  # grayscale
            return np.dstack((image, image, image, np.ones_like(image)))
        elif image.shape[2] == 3:  # RGB
            return np.dstack((image, np.ones(image.shape[:2])))
        elif image.shape[2] == 4:  # RGBA
            return image
        else:
            raise ValueError("Unsupported image format")

    def blend(self, output_dtype: np.dtype = np.float32) -> np.ndarray:
        """
        blend the foreground image and background image using the specified blend mode and alpha value

        parameters
        ------------
        output_dtype: np.dtype, optional
            the dtype of the output image, by default np.float32, can be np.uint8

        returns
        ------------
        np.ndarray
            the blended image, in RGBA format
        """
        # Blend implementation goes here
        import blendmodes.blend as bld
        from blendmodes.blendtype import BlendType

        _blend_mode: BlendType
        # Convert BlendMode to BlendType
        blend_type_map = {
            BlendMode.NORMAL: BlendType.NORMAL,
            BlendMode.SCREEN: BlendType.SCREEN,
            BlendMode.OVERLAY: BlendType.OVERLAY,
            BlendMode.MULTIPLY: BlendType.MULTIPLY,
            BlendMode.COLOR_BURN: BlendType.COLOURBURN,
            BlendMode.LINEAR_BURN: BlendType.GRAINMERGE,  # Closest equivalent
            BlendMode.DIVIDE: BlendType.DIVIDE,
            BlendMode.SOFT_LIGHT: BlendType.SOFTLIGHT,
            BlendMode.HARD_LIGHT: BlendType.HARDLIGHT,
        }
        _blend_mode = blend_type_map.get(self.blend_mode, BlendType.NORMAL)
        img_output = bld.blendLayers(
            self.background_image * 255,
            self.foreground_image * 255,
            _blend_mode,
            opacity=self.alpha,
        )
        img_output = np.asarray(img_output)

        # want to use uint8? convert output
        # if output_dtype == np.uint8:
        #     img_output = (img_output * 255).astype(np.uint8)

        return img_output
