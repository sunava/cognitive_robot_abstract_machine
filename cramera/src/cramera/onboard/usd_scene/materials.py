"""
Mapping of authored USD shaders (OmniPBR, OmniSurface, OmniGlass, UsdPreviewSurface)
onto the glTF 2.0 metallic-roughness material model.

The defaults and formulas follow NVIDIA's MDL sources (``OmniPBR_ClearCoat.mdl``,
``OmniSurfaceBase.mdl``, ``OmniGlass.mdl``) and the UsdPreviewSurface specification.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy
from PIL import Image
from scipy.integrate import quad
from typing_extensions import Self

from cramera.onboard.usd_scene.scene_description import (
    ImportedMaterial,
    MaterialSource,
    ShaderInputValue,
    TextureChannel,
    TextureFile,
)

# %% glTF material model


class AlphaMode(StrEnum):
    """
    How the alpha of the base colour is used, named as glTF names its alpha modes.
    """

    OPAQUE = "OPAQUE"
    """
    Alpha is ignored.
    """

    MASK = "MASK"
    """
    Fragments below the alpha cutoff are discarded.
    """

    BLEND = "BLEND"
    """
    The surface is blended over what lies behind it.
    """


@dataclass(frozen=True)
class TextureTransform:
    """
    An affine change of texture coordinates in glTF's convention (origin top left),
    written as ``KHR_texture_transform``.
    """

    offset: tuple[float, float] = (0.0, 0.0)
    """
    Added to the scaled coordinates.
    """

    scale: tuple[float, float] = (1.0, 1.0)
    """
    Multiplies the coordinates; values above one repeat the texture.
    """


@dataclass(frozen=True)
class ScalarChannel:
    """
    A scalar material property that blends a constant with a greyscale texture,
    ``(1 - influence) * constant + influence * texture``, as OmniPBR does.
    """

    constant: float
    """
    The value without a texture.
    """

    texture: TextureFile | None = None
    """
    The texture holding the value per texel, or None for the constant alone.
    """

    influence: float = 1.0
    """
    How much the texture replaces the constant, between 0 and 1.
    """

    @property
    def textured(self) -> bool:
        """
        Whether the texture contributes to the value.
        """
        return self.texture is not None and self.influence > 0.0

    def texel_values(self, size: tuple[int, int] | None) -> numpy.ndarray:
        """
        The blended value per texel, between 0 and 1.

        :param size: The ``(width, height)`` to resample the texture to, or None to
            keep the texture's own size.
        """
        image = Image.open(self.texture.path)
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        values = TextureChannelReader(image).values(self.texture.channel)
        return (1.0 - self.influence) * self.constant + self.influence * values

    def average(self) -> float:
        """
        The value averaged over the texels, or the constant without a texture.
        """
        if not self.textured:
            return self.constant
        return float(self.texel_values(None).mean())

    def mixed_with(self, value: float, share: float) -> ScalarChannel:
        """
        The channel blended with a constant, ``(1 - share) * channel + share * value``
        in every texel.

        :param value: The constant blended in.
        :param share: The share of the constant, between 0 and 1.
        """
        if share <= 0.0:
            return self
        if not self.textured or share >= 1.0:
            return ScalarChannel((1.0 - share) * self.constant + share * value)
        influence = (1.0 - share) * self.influence
        return ScalarChannel(
            ((1.0 - share) * (1.0 - self.influence) * self.constant + share * value)
            / (1.0 - influence),
            self.texture,
            influence,
        )


@dataclass
class TextureChannelReader:
    """
    Reads one channel of an image as floats between 0 and 1, whatever its bit depth.
    """

    image: Image.Image
    """
    The opened image.
    """

    def values(self, channel: TextureChannel) -> numpy.ndarray:
        """
        The channel's values; a single-channel image answers every channel.

        :param channel: The channel to read; for :attr:`TextureChannel.RGB` the red
            channel stands for a greyscale image stored as colour.
        """
        image = self.image.convert("RGBA") if self.image.mode == "P" else self.image
        pixels = numpy.asarray(image)
        scale = 65535.0 if image.mode.startswith("I") else 255.0
        if pixels.ndim == 2:
            return pixels.astype(numpy.float64) / scale
        index = {
            TextureChannel.RGB: 0,
            TextureChannel.RED: 0,
            TextureChannel.GREEN: 1,
            TextureChannel.BLUE: 2,
            TextureChannel.ALPHA: 3,
        }[channel]
        return (
            pixels[..., min(index, pixels.shape[-1] - 1)].astype(numpy.float64) / scale
        )


@dataclass(frozen=True)
class SrgbTransfer:
    """
    The sRGB transfer function of IEC 61966-2-1, which turns encoded colour values into
    linear light as glTF viewers decode a base colour texture.
    """

    linear_limit: float = 0.04045
    """
    The largest encoded value on the linear segment near black.
    """

    linear_slope: float = 12.92
    """
    The ratio of encoded to linear values on the linear segment.
    """

    offset: float = 0.055
    """
    The offset of the power segment.
    """

    exponent: float = 2.4
    """
    The exponent of the power segment.
    """

    def decoded(self, encoded: numpy.ndarray) -> numpy.ndarray:
        """
        The linear values of encoded values.

        :param encoded: Encoded values between 0 and 1.
        """
        encoded = numpy.asarray(encoded, dtype=numpy.float64)
        return numpy.where(
            encoded <= self.linear_limit,
            encoded / self.linear_slope,
            ((encoded + self.offset) / (1.0 + self.offset)) ** self.exponent,
        )


@dataclass(frozen=True)
class RoughnessMetallicTexture:
    """
    The glTF metallic-roughness texture packed from two greyscale sources: roughness in
    the green channel and metalness in the blue channel.
    """

    roughness: ScalarChannel
    """
    The roughness source.
    """

    metallic: ScalarChannel
    """
    The metalness source.
    """

    def encode(self) -> bytes:
        """
        The packed texture as PNG bytes, sized like the roughness texture (or the
        metalness texture when roughness is constant).
        """
        reference = self.roughness if self.roughness.textured else self.metallic
        size = Image.open(reference.texture.path).size
        green = self.channel_values(self.roughness, size)
        blue = self.channel_values(self.metallic, size)
        packed = numpy.stack(
            [numpy.full_like(green, 255), green, blue], axis=-1
        ).astype(numpy.uint8)
        stream = io.BytesIO()
        Image.fromarray(packed, mode="RGB").save(stream, format="PNG")
        return stream.getvalue()

    @staticmethod
    def channel_values(channel: ScalarChannel, size: tuple[int, int]) -> numpy.ndarray:
        """
        One channel's bytes at the packed size; a constant fills the whole image.

        :param channel: The source of the channel.
        :param size: The ``(width, height)`` of the packed texture.
        """
        if not channel.textured:
            return numpy.full((size[1], size[0]), round(channel.constant * 255.0))
        return numpy.clip(numpy.round(channel.texel_values(size) * 255.0), 0, 255)


@dataclass(frozen=True)
class PhysicallyBasedMaterial:
    """
    A glTF 2.0 metallic-roughness material with the extensions the converter writes.
    Equal materials share one glTF material and one draw call.
    """

    name: str = field(compare=False)
    """
    The name of the material on the stage.
    """

    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    """
    Linear base colour and alpha, multiplied with the base colour texture.
    """

    base_color_texture: TextureFile | None = None
    """
    The sRGB base colour texture.
    """

    metallic_factor: float = 0.0
    """
    Metalness, multiplied with the blue channel of the metallic-roughness texture.
    """

    roughness_factor: float = 0.5
    """
    Perceptual roughness, multiplied with the green channel of the
    metallic-roughness texture.
    """

    metallic_roughness_texture: RoughnessMetallicTexture | None = None
    """
    The packed roughness and metalness texture.
    """

    normal_texture: TextureFile | None = None
    """
    The tangent-space normal map.
    """

    normal_scale: float = 1.0
    """
    The strength of the normal map.
    """

    emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """
    Linear emitted colour, each component at most one.
    """

    emissive_texture: TextureFile | None = None
    """
    The sRGB emission texture.
    """

    emissive_strength: float = 1.0
    """
    Multiplies the emission beyond the unit range (``KHR_materials_emissive_strength``).
    """

    alpha_mode: AlphaMode = AlphaMode.OPAQUE
    """
    How alpha is used.
    """

    alpha_cutoff: float = 0.5
    """
    The alpha threshold in :attr:`AlphaMode.MASK`.
    """

    double_sided: bool = False
    """
    Whether back faces are drawn.
    """

    transmission: float = 0.0
    """
    The share of light passing through the surface, tinted by the base colour.
    """

    index_of_refraction: float = 1.5
    """
    The index of refraction (``KHR_materials_ior``).
    """

    clearcoat: float = 0.0
    """
    The strength of a clear coat layer (``KHR_materials_clearcoat``).
    """

    clearcoat_roughness: float = 0.0
    """
    The roughness of the clear coat layer.
    """

    anisotropy: float = 0.0
    """
    The strength of anisotropic reflection (``KHR_materials_anisotropy``).
    """

    texture_transform: TextureTransform | None = None
    """
    A change of texture coordinates shared by every texture of the material.
    """

    unmapped_inputs: tuple[str, ...] = field(default=(), compare=False)
    """
    The authored shader inputs the mapping did not carry over.
    """

    @property
    def transmissive(self) -> bool:
        """
        Whether light passes through the surface.
        """
        return self.transmission > 0.0

    @property
    def textured(self) -> bool:
        """
        Whether any texture of the material reads texture coordinates.
        """
        return any(
            texture is not None
            for texture in (
                self.base_color_texture,
                self.metallic_roughness_texture,
                self.normal_texture,
                self.emissive_texture,
            )
        )

    def with_double_sides(self) -> Self:
        """
        The same material drawn from both sides.
        """
        return replace(self, double_sided=True)


# %% transmission drawn by blending


@dataclass
class TransmissionBlending:
    """
    Approximates a transmissive surface by an alpha-blended one, for viewers that
    cannot refract. Each drawn face stands for one interface of the material.

    Its alpha covers all but the light behind it that the interface passes along the
    line of sight in every colour channel, ``weight * (1 - R) * min(tint) * (1 -
    scattering * frosting)``, where ``R`` is the interface's reflectance averaged over
    all directions, since one alpha holds for every view. The tint is the base colour
    factor times the average colour of the base colour texture, and the frosting is
    the roughness averaged over the surface. Its colour is chosen so that, in front of
    surroundings as bright as what it reflects, the blend shows what the interface
    sends towards the eye: the reflection ``R`` plus the tinted rest ``(1 - R) *
    tint``. The transmitted share of the surface reflects without diffusing, so it is
    drawn metallic.

    .. note:: A closed solid such as a pane is drawn with its front and back faces, so
        the light behind it is covered twice, as it crosses two interfaces; a
        thin-walled plane crosses one. Volume absorption depends on the path length
        through the solid, which a material does not know, and is not drawn.

    .. note:: A base colour texture keeps multiplying the drawn colour, so the colour
        is exact at the texture's average colour; where that would need a factor above
        one, the factor is one.
    """

    frosted_scattering: float = 1.0
    """
    The share of the transmitted light that a fully frosted surface (roughness one)
    scatters out of the line of sight; less frosting scatters proportionally less.
    """

    transfer: SrgbTransfer = field(default_factory=SrgbTransfer)
    """
    The decoding of base colour textures into linear light.
    """

    def approximate(self, material: PhysicallyBasedMaterial) -> PhysicallyBasedMaterial:
        """
        The blended, double-sided material drawn in place of a transmissive one; any
        other material is drawn as it is.

        :param material: The material as authored, its base colour tinting the
            transmitted light and its roughness frosting it.
        """
        if not material.transmissive:
            return material
        texture_color = self.texture_color(material.base_color_texture)
        tint = numpy.asarray(material.base_color_factor[:3]) * texture_color
        reflectance = self.reflectance(material.index_of_refraction)
        see_through = self.see_through(material, tint)
        shown = (
            tint
            if see_through >= 1.0
            else (reflectance + (1.0 - reflectance) * tint - see_through)
            / (1.0 - see_through)
        )
        factor = numpy.minimum(
            numpy.divide(
                shown, texture_color, out=numpy.zeros(3), where=texture_color > 0.0
            ),
            1.0,
        )
        return self.reflecting(
            replace(
                material,
                base_color_factor=(
                    *(float(component) for component in factor),
                    material.base_color_factor[3] * (1.0 - see_through),
                ),
                alpha_mode=AlphaMode.BLEND,
                double_sided=True,
            )
        )

    def see_through(
        self, material: PhysicallyBasedMaterial, tint: numpy.ndarray
    ) -> float:
        """
        The share of the light behind one face of a transmissive material that the
        face passes along the line of sight.

        :param material: The transmissive material.
        :param tint: The linear colour the material tints the transmitted light with.
        """
        return float(
            material.transmission
            * (1.0 - self.reflectance(material.index_of_refraction))
            * numpy.min(tint)
            * (1.0 - self.frosted_scattering * self.frosting(material))
        )

    @staticmethod
    def frosting(material: PhysicallyBasedMaterial) -> float:
        """
        The roughness of a material averaged over its surface: the roughness factor
        times the average of the roughness texture.

        :param material: The material.
        """
        texture = material.metallic_roughness_texture
        if texture is None:
            return material.roughness_factor
        return material.roughness_factor * texture.roughness.average()

    def texture_color(self, texture: TextureFile | None) -> numpy.ndarray:
        """
        The average linear colour of a base colour texture, decoded from sRGB as glTF
        viewers decode it, or white without a texture.

        :param texture: The base colour texture.
        """
        if texture is None:
            return numpy.ones(3)
        reader = TextureChannelReader(Image.open(texture.path))
        encoded = numpy.stack(
            [
                reader.values(channel)
                for channel in (
                    TextureChannel.RED,
                    TextureChannel.GREEN,
                    TextureChannel.BLUE,
                )
            ]
        )
        return self.transfer.decoded(encoded).reshape(3, -1).mean(axis=1)

    @staticmethod
    def reflecting(material: PhysicallyBasedMaterial) -> PhysicallyBasedMaterial:
        """
        The material with the transmitted share of its surface metallic: every
        metalness ``m`` becomes ``m + transmission * (1 - m)``.

        :param material: The transmissive material.
        """
        share = material.transmission
        texture = material.metallic_roughness_texture
        if texture is None:
            return replace(
                material,
                metallic_factor=ScalarChannel(material.metallic_factor)
                .mixed_with(1.0, share)
                .constant,
            )
        metallic = texture.metallic.mixed_with(
            0.0, 1.0 - material.metallic_factor
        ).mixed_with(1.0, share)
        if not texture.roughness.textured and not metallic.textured:
            return replace(
                material,
                roughness_factor=material.roughness_factor * texture.roughness.constant,
                metallic_factor=metallic.constant,
                metallic_roughness_texture=None,
            )
        return replace(
            material,
            metallic_factor=1.0,
            metallic_roughness_texture=replace(texture, metallic=metallic),
        )

    @classmethod
    def reflectance(cls, index_of_refraction: float) -> float:
        """
        The reflectance of an interface between air and the material for light from
        all directions: the Fresnel reflectance averaged over the cosine-weighted
        hemisphere of directions of incidence.

        :param index_of_refraction: The index of refraction of the material.
        """
        average, _ = quad(
            lambda cosine: 2.0
            * cosine
            * cls.fresnel_reflectance(cosine, index_of_refraction),
            0.0,
            1.0,
        )
        return average

    @staticmethod
    def fresnel_reflectance(cosine: float, index_of_refraction: float) -> float:
        """
        The share of unpolarized light from air that an interface with the material
        reflects (Fresnel equations).

        :param cosine: The cosine of the angle of incidence, above zero.
        :param index_of_refraction: The index of refraction of the material.
        """
        refracted_sine_squared = (1.0 - cosine**2) / index_of_refraction**2
        refracted_cosine = numpy.sqrt(max(0.0, 1.0 - refracted_sine_squared))
        perpendicular = (cosine - index_of_refraction * refracted_cosine) / (
            cosine + index_of_refraction * refracted_cosine
        )
        parallel = (index_of_refraction * cosine - refracted_cosine) / (
            index_of_refraction * cosine + refracted_cosine
        )
        return float(perpendicular**2 + parallel**2) / 2.0


# %% shader inputs


@dataclass(frozen=True)
class TexturedColor:
    """
    A linear colour factor multiplied with an optional sRGB colour texture, as glTF
    gives a base colour.
    """

    factor: tuple[float, float, float]
    """
    The linear colour, multiplying the texture when there is one.
    """

    texture: TextureFile | None = None
    """
    The colour texture, or None for the factor alone.
    """

    def scaled(self, scale: float) -> TexturedColor:
        """
        The colour with its factor multiplied.

        :param scale: The multiplier.
        """
        return replace(
            self, factor=tuple(component * scale for component in self.factor)
        )

    def mixed_with(self, other: TexturedColor, share: float) -> TexturedColor:
        """
        The colour blended with another. Only one texture can be drawn, so the blend
        keeps the texture of the colour with the larger share.

        :param other: The colour blended in.
        :param share: The share of the other colour, between 0 and 1.
        """
        return TexturedColor(
            tuple(
                (1.0 - share) * own + share * blended
                for own, blended in zip(self.factor, other.factor)
            ),
            other.texture if share > 1.0 - share else self.texture,
        )


@dataclass
class ShaderInputs:
    """
    Typed access to a material's authored inputs, keeping track of what was read.
    """

    material: ImportedMaterial
    """
    The material whose inputs are read.
    """

    read: set[str] = field(default_factory=set)
    """
    The names of the inputs a mapping asked for.
    """

    def number(self, name: str, default: float) -> float:
        """
        A scalar input.

        :param name: The input name.
        :param default: The shader's default for the input.
        """
        self.read.add(name)
        value = self.material.inputs.get(name, default)
        return float(value) if isinstance(value, (bool, int, float)) else default

    def flag(self, name: str, default: bool) -> bool:
        """
        A boolean input.

        :param name: The input name.
        :param default: The shader's default for the input.
        """
        return bool(self.number(name, float(default)))

    def color(
        self, name: str, default: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """
        A colour input.

        :param name: The input name.
        :param default: The shader's default for the input.
        """
        self.read.add(name)
        value = self.material.inputs.get(name, default)
        if not isinstance(value, tuple) or len(value) < 3:
            return default
        return float(value[0]), float(value[1]), float(value[2])

    def vector(self, name: str, default: tuple[float, float]) -> tuple[float, float]:
        """
        A two-component input.

        :param name: The input name.
        :param default: The shader's default for the input.
        """
        self.read.add(name)
        value = self.material.inputs.get(name, default)
        if not isinstance(value, tuple) or len(value) < 2:
            return default
        return float(value[0]), float(value[1])

    def texture(self, name: str) -> TextureFile | None:
        """
        A texture input, or None when unset or unresolved.

        :param name: The input name.
        """
        self.read.add(name)
        value = self.material.inputs.get(name)
        return value if isinstance(value, TextureFile) else None

    def textured_color(
        self,
        color_name: str,
        texture_name: str,
        default: tuple[float, float, float],
    ) -> TexturedColor:
        """
        A colour input whose texture, when one is set, replaces the constant.

        :param color_name: The name of the constant colour input.
        :param texture_name: The name of the texture input.
        :param default: The shader's default for the constant.
        """
        texture = self.texture(texture_name)
        if texture is not None:
            return TexturedColor((1.0, 1.0, 1.0), texture)
        return TexturedColor(self.color(color_name, default))

    def unmapped(self, neutral: dict[str, ShaderInputValue]) -> tuple[str, ...]:
        """
        The authored inputs that were never read and differ from their neutral value.

        :param neutral: Inputs the mapping may ignore while they hold these values.
        """
        return tuple(
            sorted(
                name
                for name, value in self.material.inputs.items()
                if name not in self.read
                and not (name in neutral and neutral[name] == value)
            )
        )


# %% shader mappings


@dataclass
class ShaderMapping(ABC):
    """
    Maps the inputs of one shading model onto a glTF material.
    """

    neutral_inputs: dict[str, ShaderInputValue] = field(default_factory=dict)
    """
    Inputs the glTF model cannot express but that are harmless at these values.
    """

    def map(self, material: ImportedMaterial) -> PhysicallyBasedMaterial:
        """
        The glTF material for an authored material, listing the inputs left out.

        :param material: The authored material.
        """
        inputs = ShaderInputs(material)
        mapped = self.map_inputs(inputs)
        return replace(
            mapped,
            name=material.name,
            unmapped_inputs=inputs.unmapped(self.neutral_inputs),
        )

    @abstractmethod
    def map_inputs(self, inputs: ShaderInputs) -> PhysicallyBasedMaterial:
        """
        The glTF material for the inputs; the name is filled in by :meth:`map`.

        :param inputs: The authored inputs.
        """


class OmniPhysicallyBasedInput(StrEnum):
    """
    The OmniPBR inputs the mapping knows: those it reads, and those glTF cannot express
    that it may leave out at a neutral value.
    """

    DIFFUSE_COLOR = "diffuse_color_constant"
    """
    The albedo without a texture.
    """

    DIFFUSE_TEXTURE = "diffuse_texture"
    """
    The albedo texture.
    """

    DIFFUSE_TINT = "diffuse_tint"
    """
    Multiplies the albedo.
    """

    ROUGHNESS = "reflection_roughness_constant"
    """
    The roughness without a texture.
    """

    ROUGHNESS_TEXTURE = "reflectionroughness_texture"
    """
    The roughness texture.
    """

    ROUGHNESS_INFLUENCE = "reflection_roughness_texture_influence"
    """
    How much the roughness texture replaces the constant.
    """

    METALLIC = "metallic_constant"
    """
    The metalness without a texture.
    """

    METALLIC_TEXTURE = "metallic_texture"
    """
    The metalness texture.
    """

    METALLIC_INFLUENCE = "metallic_texture_influence"
    """
    How much the metalness texture replaces the constant.
    """

    ENABLE_PACKED_TEXTURE = "enable_ORM_texture"
    """
    Whether occlusion, roughness and metalness come from one packed texture.
    """

    PACKED_TEXTURE = "ORM_texture"
    """
    The packed occlusion, roughness and metalness texture.
    """

    NORMAL_TEXTURE = "normalmap_texture"
    """
    The normal map.
    """

    BUMP_FACTOR = "bump_factor"
    """
    The strength of the normal map.
    """

    ENABLE_EMISSION = "enable_emission"
    """
    Whether the surface emits light.
    """

    EMISSIVE_COLOR = "emissive_color"
    """
    The emitted colour without a texture.
    """

    EMISSIVE_TEXTURE = "emissive_color_texture"
    """
    The emitted colour texture.
    """

    EMISSIVE_INTENSITY = "emissive_intensity"
    """
    The emitted luminance in nits.
    """

    ENABLE_OPACITY = "enable_opacity"
    """
    Whether the surface is partly transparent.
    """

    OPACITY = "opacity_constant"
    """
    The opacity without a texture.
    """

    OPACITY_THRESHOLD = "opacity_threshold"
    """
    The opacity below which the surface is cut out.
    """

    TEXTURE_TRANSLATE = "texture_translate"
    """
    Shifts the texture coordinates.
    """

    TEXTURE_SCALE = "texture_scale"
    """
    Repeats the texture.
    """

    SPECULAR_LEVEL = "specular_level"
    """
    The strength of the specular reflection.
    """

    PROJECT_COORDINATES = "project_uvw"
    """
    Whether texture coordinates are projected instead of read from the mesh.
    """

    WORLD_OR_OBJECT = "world_or_object"
    """
    Whether projected coordinates are in world rather than object space.
    """

    ALBEDO_DESATURATION = "albedo_desaturation"
    """
    How much the albedo is desaturated.
    """

    ALBEDO_ADD = "albedo_add"
    """
    A value added to the albedo.
    """

    ALBEDO_BRIGHTNESS = "albedo_brightness"
    """
    Scales the albedo.
    """

    TEXTURE_ROTATE = "texture_rotate"
    """
    Turns the texture coordinates.
    """

    EXCLUDE_FROM_WHITE_MODE = "excludeFromWhiteMode"
    """
    Whether the material keeps its look in the renderer's white mode.
    """


@dataclass(frozen=True)
class Emission:
    """
    Emitted light as a glTF material describes it.
    """

    factor: tuple[float, float, float]
    """
    The linear emissive colour, each component at most one.
    """

    strength: float
    """
    The factor scaling the emissive colour, ``KHR_materials_emissive_strength``.
    """


@dataclass
class EmissionScale:
    """
    Converts an MDL emission intensity (in nits) into glTF's emissive factor and
    ``KHR_materials_emissive_strength``: the emitted colour is the authored colour
    times ``intensity / reference``, with the part above one moved into the strength.
    """

    reference_intensity: float = 1000.0
    """
    The MDL emission intensity shown as the plain emissive colour.
    """

    def emission(self, color: tuple[float, float, float], intensity: float) -> Emission:
        """
        The glTF emission of an authored colour and intensity.

        :param color: The authored linear emission colour.
        :param intensity: The authored emission intensity.
        """
        ratio = max(intensity, 0.0) / self.reference_intensity
        brightest = max(max(color), 1e-9)
        scale = min(ratio, 1.0 / brightest)
        return Emission(
            tuple(min(component * scale, 1.0) for component in color),
            max(ratio / scale, 1.0) if scale > 0.0 else 1.0,
        )


@dataclass
class OmniPhysicallyBasedMapping(ShaderMapping):
    """
    OmniPBR: the base colour is the diffuse texture (or constant) times the tint;
    roughness and metalness blend constant and texture by their influence.
    """

    diffuse_color: tuple[float, float, float] = (0.2, 0.2, 0.2)
    """
    OmniPBR's default ``diffuse_color_constant``.
    """

    roughness: float = 0.5
    """
    OmniPBR's default ``reflection_roughness_constant``.
    """

    emissive_color: tuple[float, float, float] = (1.0, 0.1, 0.1)
    """
    OmniPBR's default ``emissive_color``.
    """

    emissive_intensity: float = 40.0
    """
    OmniPBR's default ``emissive_intensity``.
    """

    emission: EmissionScale = field(default_factory=EmissionScale)
    """
    The conversion of emission intensities.
    """

    neutral_inputs: dict[str, ShaderInputValue] = field(
        default_factory=lambda: {
            OmniPhysicallyBasedInput.SPECULAR_LEVEL: 0.5,
            OmniPhysicallyBasedInput.PROJECT_COORDINATES: False,
            OmniPhysicallyBasedInput.WORLD_OR_OBJECT: False,
            OmniPhysicallyBasedInput.ALBEDO_DESATURATION: 0.0,
            OmniPhysicallyBasedInput.ALBEDO_ADD: 0.0,
            OmniPhysicallyBasedInput.ALBEDO_BRIGHTNESS: 1.0,
            OmniPhysicallyBasedInput.TEXTURE_ROTATE: 0.0,
            OmniPhysicallyBasedInput.EXCLUDE_FROM_WHITE_MODE: False,
        }
    )
    """
    Inputs the glTF model cannot express but that are harmless at these values.
    """

    def map_inputs(self, inputs: ShaderInputs) -> PhysicallyBasedMaterial:
        """
        The glTF material for the inputs; the name is filled in by :meth:`map`.

        :param inputs: The authored inputs.
        """
        input_name = OmniPhysicallyBasedInput
        tint = inputs.color(input_name.DIFFUSE_TINT, (1.0, 1.0, 1.0))
        diffuse_texture = inputs.texture(input_name.DIFFUSE_TEXTURE)
        diffuse = (
            (1.0, 1.0, 1.0)
            if diffuse_texture is not None
            else inputs.color(input_name.DIFFUSE_COLOR, self.diffuse_color)
        )
        packed = (
            inputs.texture(input_name.PACKED_TEXTURE)
            if inputs.flag(input_name.ENABLE_PACKED_TEXTURE, False)
            else None
        )
        roughness = ScalarChannel(
            constant=inputs.number(input_name.ROUGHNESS, self.roughness),
            texture=(
                replace(packed, channel=TextureChannel.GREEN)
                if packed is not None
                else inputs.texture(input_name.ROUGHNESS_TEXTURE)
            ),
            influence=inputs.number(input_name.ROUGHNESS_INFLUENCE, 0.0),
        )
        metallic = ScalarChannel(
            constant=inputs.number(input_name.METALLIC, 0.0),
            texture=(
                replace(packed, channel=TextureChannel.BLUE)
                if packed is not None
                else inputs.texture(input_name.METALLIC_TEXTURE)
            ),
            influence=inputs.number(input_name.METALLIC_INFLUENCE, 0.0),
        )
        enable_opacity = inputs.flag(input_name.ENABLE_OPACITY, False)
        opacity = inputs.number(input_name.OPACITY, 1.0) if enable_opacity else 1.0
        threshold = inputs.number(input_name.OPACITY_THRESHOLD, 0.0)
        material = PhysicallyBasedMaterial(
            name="",
            base_color_factor=(
                *(
                    diffuse_channel * tint_channel
                    for diffuse_channel, tint_channel in zip(diffuse, tint)
                ),
                opacity,
            ),
            base_color_texture=diffuse_texture,
            normal_texture=inputs.texture(input_name.NORMAL_TEXTURE),
            normal_scale=inputs.number(input_name.BUMP_FACTOR, 1.0),
            alpha_mode=(
                AlphaMode.OPAQUE
                if not enable_opacity
                else (AlphaMode.MASK if threshold > 0.0 else AlphaMode.BLEND)
            ),
            alpha_cutoff=threshold if threshold > 0.0 else 0.5,
            texture_transform=self.texture_transform(inputs),
        )
        material = RoughnessMetallicPacking(roughness, metallic).apply(material)
        return self.with_emission(material, inputs)

    def with_emission(
        self, material: PhysicallyBasedMaterial, inputs: ShaderInputs
    ) -> PhysicallyBasedMaterial:
        """
        Add OmniPBR's emission to a material.

        :param material: The material without emission.
        :param inputs: The authored inputs.
        """
        input_name = OmniPhysicallyBasedInput
        enabled = inputs.flag(input_name.ENABLE_EMISSION, False)
        texture = inputs.texture(input_name.EMISSIVE_TEXTURE)
        color = inputs.color(input_name.EMISSIVE_COLOR, self.emissive_color)
        intensity = inputs.number(
            input_name.EMISSIVE_INTENSITY, self.emissive_intensity
        )
        if not enabled or intensity <= 0.0:
            return material
        emission = self.emission.emission(
            (1.0, 1.0, 1.0) if texture is not None else color, intensity
        )
        return replace(
            material,
            emissive_factor=emission.factor,
            emissive_texture=texture,
            emissive_strength=emission.strength,
        )

    @staticmethod
    def texture_transform(inputs: ShaderInputs) -> TextureTransform | None:
        """
        OmniPBR's tiling, ``uv * scale + translate``, rewritten for glTF's texture
        coordinates whose V axis is flipped (``v' = 1 - v``).

        :param inputs: The authored inputs.
        """
        scale = inputs.vector(OmniPhysicallyBasedInput.TEXTURE_SCALE, (1.0, 1.0))
        translate = inputs.vector(
            OmniPhysicallyBasedInput.TEXTURE_TRANSLATE, (0.0, 0.0)
        )
        if scale == (1.0, 1.0) and translate == (0.0, 0.0):
            return None
        return TextureTransform(
            offset=(translate[0], 1.0 - scale[1] - translate[1]), scale=scale
        )


@dataclass
class RoughnessMetallicPacking:
    """
    Puts roughness and metalness into a glTF material: as factors while both are
    constant, else as one packed texture with the blend baked in.
    """

    roughness: ScalarChannel
    """
    The roughness source.
    """

    metallic: ScalarChannel
    """
    The metalness source.
    """

    def apply(self, material: PhysicallyBasedMaterial) -> PhysicallyBasedMaterial:
        """
        The material with roughness and metalness set.

        :param material: The material to complete.
        """
        if not self.roughness.textured and not self.metallic.textured:
            return replace(
                material,
                roughness_factor=self.roughness.constant,
                metallic_factor=self.metallic.constant,
            )
        return replace(
            material,
            roughness_factor=1.0,
            metallic_factor=1.0,
            metallic_roughness_texture=RoughnessMetallicTexture(
                self.roughness, self.metallic
            ),
        )


class OmniSurfaceInput(StrEnum):
    """
    The OmniSurface inputs the mapping knows: those it reads, and those glTF cannot
    express that it may leave out at a neutral value.
    """

    DIFFUSE_WEIGHT = "diffuse_reflection_weight"
    """
    Scales the diffuse colour.
    """

    DIFFUSE_COLOR = "diffuse_reflection_color"
    """
    The diffuse colour.
    """

    DIFFUSE_IMAGE = "diffuse_reflection_color_image"
    """
    The diffuse colour texture.
    """

    METALNESS = "metalness"
    """
    The metalness.
    """

    METALNESS_IMAGE = "metalness_image"
    """
    The metalness texture.
    """

    ROUGHNESS = "specular_reflection_roughness"
    """
    The specular roughness.
    """

    ROUGHNESS_IMAGE = "specular_reflection_roughness_image"
    """
    The specular roughness texture.
    """

    ANISOTROPY = "specular_reflection_anisotropy"
    """
    The anisotropy of the specular reflection.
    """

    INDEX_OF_REFRACTION = "specular_reflection_ior"
    """
    The index of refraction of the specular layer.
    """

    COAT_WEIGHT = "coat_weight"
    """
    The strength of the coat layer.
    """

    COAT_ROUGHNESS = "coat_roughness"
    """
    The roughness of the coat layer.
    """

    ENABLE_TRANSMISSION = "enable_specular_transmission"
    """
    Whether light passes through the surface.
    """

    TRANSMISSION_WEIGHT = "specular_transmission_weight"
    """
    The share of light passing through.
    """

    TRANSMISSION_COLOR = "specular_transmission_color"
    """
    The tint of the light passing through.
    """

    TRANSMISSION_COLOR_IMAGE = "specular_transmission_color_image"
    """
    The tint texture of the light passing through.
    """

    EMISSION_WEIGHT = "emission_weight"
    """
    The strength of the emission.
    """

    EMISSION_COLOR = "emission_color"
    """
    The emitted colour.
    """

    EMISSION_INTENSITY = "emission_intensity"
    """
    The emitted luminance.
    """

    NORMAL_IMAGE = "geometry_normal_image"
    """
    The normal map.
    """

    ENABLE_OPACITY = "enable_opacity"
    """
    Whether the surface is partly transparent.
    """

    OPACITY = "geometry_opacity"
    """
    The opacity.
    """

    DIFFUSE_ROUGHNESS = "diffuse_reflection_roughness"
    """
    The roughness of the diffuse reflection.
    """

    SPECULAR_WEIGHT = "specular_reflection_weight"
    """
    The strength of the specular reflection.
    """

    SPECULAR_COLOR = "specular_reflection_color"
    """
    The tint of the specular reflection.
    """

    ANISOTROPY_ROTATION = "specular_reflection_anisotropy_rotation"
    """
    Turns the direction of the anisotropic reflection.
    """

    SUBSURFACE_WEIGHT = "subsurface_weight"
    """
    The share of light scattered below the surface.
    """

    COAT_COLOR = "coat_color"
    """
    The tint of the coat layer.
    """

    COAT_INDEX_OF_REFRACTION = "coat_ior"
    """
    The index of refraction of the coat layer.
    """

    THIN_WALLED = "thin_walled"
    """
    Whether the surface is a thin shell without an inside.
    """


@dataclass
class OmniSurfaceMapping(ShaderMapping):
    """
    OmniSurface: the base colour is the diffuse colour (or image) times the diffuse
    weight, mixed with the transmission colour (or image) by the transmission weight;
    images replace their constants, and of two images the one with the larger weight is
    kept.
    """

    diffuse_weight: float = 0.8
    """
    OmniSurface's default ``diffuse_reflection_weight``.
    """

    diffuse_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """
    OmniSurface's default ``diffuse_reflection_color``.
    """

    transmission_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """
    OmniSurface's default ``specular_transmission_color``.
    """

    roughness: float = 0.2
    """
    OmniSurface's default ``specular_reflection_roughness``.
    """

    coat_roughness: float = 0.1
    """
    OmniSurface's default ``coat_roughness``.
    """

    emission: EmissionScale = field(default_factory=EmissionScale)
    """
    The conversion of emission intensities.
    """

    neutral_inputs: dict[str, ShaderInputValue] = field(
        default_factory=lambda: {
            OmniSurfaceInput.DIFFUSE_ROUGHNESS: 0.0,
            OmniSurfaceInput.SPECULAR_WEIGHT: 1.0,
            OmniSurfaceInput.SPECULAR_COLOR: (1.0, 1.0, 1.0),
            OmniSurfaceInput.ANISOTROPY_ROTATION: 0.0,
            OmniSurfaceInput.SUBSURFACE_WEIGHT: 0.0,
            OmniSurfaceInput.COAT_COLOR: (1.0, 1.0, 1.0),
            OmniSurfaceInput.COAT_INDEX_OF_REFRACTION: 1.5,
            OmniSurfaceInput.THIN_WALLED: False,
        }
    )
    """
    Inputs the glTF model cannot express but that are harmless at these values.
    """

    def map_inputs(self, inputs: ShaderInputs) -> PhysicallyBasedMaterial:
        """
        The glTF material for the inputs; the name is filled in by :meth:`map`.

        :param inputs: The authored inputs.
        """
        input_name = OmniSurfaceInput
        diffuse = inputs.textured_color(
            input_name.DIFFUSE_COLOR, input_name.DIFFUSE_IMAGE, self.diffuse_color
        ).scaled(inputs.number(input_name.DIFFUSE_WEIGHT, self.diffuse_weight))
        transmission = (
            inputs.number(input_name.TRANSMISSION_WEIGHT, 0.0)
            if inputs.flag(input_name.ENABLE_TRANSMISSION, False)
            else 0.0
        )
        color = (
            diffuse
            if transmission <= 0.0
            else diffuse.mixed_with(
                inputs.textured_color(
                    input_name.TRANSMISSION_COLOR,
                    input_name.TRANSMISSION_COLOR_IMAGE,
                    self.transmission_color,
                ),
                transmission,
            )
        )
        opacity = (
            inputs.number(input_name.OPACITY, 1.0)
            if inputs.flag(input_name.ENABLE_OPACITY, False)
            else 1.0
        )
        material = PhysicallyBasedMaterial(
            name="",
            base_color_factor=(*color.factor, opacity),
            base_color_texture=color.texture,
            normal_texture=inputs.texture(input_name.NORMAL_IMAGE),
            index_of_refraction=inputs.number(input_name.INDEX_OF_REFRACTION, 1.5),
            clearcoat=inputs.number(input_name.COAT_WEIGHT, 0.0),
            clearcoat_roughness=inputs.number(
                input_name.COAT_ROUGHNESS, self.coat_roughness
            ),
            anisotropy=inputs.number(input_name.ANISOTROPY, 0.0),
            transmission=transmission,
            alpha_mode=(
                AlphaMode.BLEND
                if transmission > 0.0 or opacity < 1.0
                else AlphaMode.OPAQUE
            ),
        )
        material = RoughnessMetallicPacking(
            ScalarChannel(
                inputs.number(input_name.ROUGHNESS, self.roughness),
                inputs.texture(input_name.ROUGHNESS_IMAGE),
            ),
            ScalarChannel(
                inputs.number(input_name.METALNESS, 0.0),
                inputs.texture(input_name.METALNESS_IMAGE),
            ),
        ).apply(material)
        emission = inputs.number(input_name.EMISSION_WEIGHT, 0.0)
        if emission <= 0.0:
            return material
        emission_color = inputs.color(input_name.EMISSION_COLOR, (1.0, 1.0, 1.0))
        emitted = self.emission.emission(
            tuple(component * emission for component in emission_color),
            inputs.number(input_name.EMISSION_INTENSITY, 1.0),
        )
        return replace(
            material,
            emissive_factor=emitted.factor,
            emissive_strength=emitted.strength,
        )


class OmniGlassInput(StrEnum):
    """
    The OmniGlass inputs the mapping knows: those it reads, and those glTF cannot
    express that it may leave out at a neutral value.
    """

    COLOR = "glass_color"
    """
    The tint of the glass.
    """

    COLOR_TEXTURE = "glass_color_texture"
    """
    The tint texture.
    """

    USE_COLOR_TEXTURE = "use_glass_color_texture"
    """
    A hidden switch OmniGlass V2 ignores: it uses the tint texture whenever one is set.
    """

    INDEX_OF_REFRACTION = "glass_ior"
    """
    The index of refraction.
    """

    FROSTING = "frosting_roughness"
    """
    The roughness of the glass.
    """

    ROUGHNESS_TEXTURE = "roughness_texture"
    """
    The roughness texture.
    """

    ROUGHNESS_INFLUENCE = "roughness_texture_influence"
    """
    How much the roughness texture replaces the frosting.
    """

    NORMAL_TEXTURE = "normal_map_texture"
    """
    The normal map.
    """

    NORMAL_STRENGTH = "normal_map_strength"
    """
    The strength of the normal map.
    """

    THIN_WALLED = "thin_walled"
    """
    Whether the glass is a thin shell without an inside.
    """

    DEPTH = "depth"
    """
    The absorption coefficient of the glass volume.
    """

    REFLECTION_COLOR = "reflection_color"
    """
    The tint of the reflection.
    """

    ENABLE_OPACITY = "enable_opacity"
    """
    Whether the surface is cut out by its opacity.
    """

    CUTOUT_OPACITY = "cutout_opacity"
    """
    The opacity of the cut out.
    """

    TEXTURE_ROTATE = "texture_rotate"
    """
    Turns the texture coordinates.
    """


@dataclass
class OmniGlassMapping(ShaderMapping):
    """
    OmniGlass: a fully transmissive, blended, double-sided surface tinted with the
    glass colour, or with the glass colour texture which replaces it; frosting becomes
    roughness.
    """

    index_of_refraction: float = 1.491
    """
    OmniGlass's default ``glass_ior``.
    """

    glass_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """
    OmniGlass's default ``glass_color``.
    """

    neutral_inputs: dict[str, ShaderInputValue] = field(
        default_factory=lambda: {
            OmniGlassInput.USE_COLOR_TEXTURE: True,
            OmniGlassInput.THIN_WALLED: False,
            OmniGlassInput.DEPTH: 0.001,
            OmniGlassInput.REFLECTION_COLOR: (1.0, 1.0, 1.0),
            OmniGlassInput.ENABLE_OPACITY: False,
            OmniGlassInput.CUTOUT_OPACITY: 1.0,
            OmniGlassInput.TEXTURE_ROTATE: 0.0,
        }
    )
    """
    Inputs the glTF model cannot express but that are harmless at these values.
    """

    def map_inputs(self, inputs: ShaderInputs) -> PhysicallyBasedMaterial:
        """
        The glTF material for the inputs; the name is filled in by :meth:`map`.

        :param inputs: The authored inputs.
        """
        input_name = OmniGlassInput
        color = inputs.textured_color(
            input_name.COLOR, input_name.COLOR_TEXTURE, self.glass_color
        )
        return RoughnessMetallicPacking(
            ScalarChannel(
                inputs.number(input_name.FROSTING, 0.0),
                inputs.texture(input_name.ROUGHNESS_TEXTURE),
                inputs.number(input_name.ROUGHNESS_INFLUENCE, 1.0),
            ),
            ScalarChannel(0.0),
        ).apply(
            PhysicallyBasedMaterial(
                name="",
                base_color_factor=(*color.factor, 1.0),
                base_color_texture=color.texture,
                normal_texture=inputs.texture(input_name.NORMAL_TEXTURE),
                normal_scale=inputs.number(input_name.NORMAL_STRENGTH, 1.0),
                transmission=1.0,
                index_of_refraction=inputs.number(
                    input_name.INDEX_OF_REFRACTION, self.index_of_refraction
                ),
                alpha_mode=AlphaMode.BLEND,
                double_sided=True,
            )
        )


class PreviewSurfaceInput(StrEnum):
    """
    The UsdPreviewSurface inputs the mapping knows: those it reads, and those glTF
    cannot express that it may leave out at a neutral value.
    """

    DIFFUSE_COLOR = "diffuseColor"
    """
    The base colour, or a texture.
    """

    EMISSIVE_COLOR = "emissiveColor"
    """
    The emitted colour, or a texture.
    """

    METALLIC = "metallic"
    """
    The metalness, or a texture.
    """

    ROUGHNESS = "roughness"
    """
    The roughness, or a texture.
    """

    OPACITY = "opacity"
    """
    The opacity.
    """

    OPACITY_THRESHOLD = "opacityThreshold"
    """
    The opacity below which the surface is cut out.
    """

    INDEX_OF_REFRACTION = "ior"
    """
    The index of refraction.
    """

    NORMAL = "normal"
    """
    The normal map.
    """

    CLEARCOAT = "clearcoat"
    """
    The strength of the clear coat.
    """

    CLEARCOAT_ROUGHNESS = "clearcoatRoughness"
    """
    The roughness of the clear coat.
    """

    USE_SPECULAR_WORKFLOW = "useSpecularWorkflow"
    """
    Whether the specular colour replaces the metalness.
    """

    SPECULAR_COLOR = "specularColor"
    """
    The specular colour of the specular workflow.
    """

    DISPLACEMENT = "displacement"
    """
    The displacement of the surface.
    """

    OCCLUSION = "occlusion"
    """
    The ambient occlusion.
    """


@dataclass
class PreviewSurfaceMapping(ShaderMapping):
    """
    UsdPreviewSurface, whose metallic workflow is glTF's model; a connected texture
    replaces the constant of its input.
    """

    diffuse_color: tuple[float, float, float] = (0.18, 0.18, 0.18)
    """
    UsdPreviewSurface's default ``diffuseColor``.
    """

    neutral_inputs: dict[str, ShaderInputValue] = field(
        default_factory=lambda: {
            PreviewSurfaceInput.USE_SPECULAR_WORKFLOW: 0,
            PreviewSurfaceInput.SPECULAR_COLOR: (0.0, 0.0, 0.0),
            PreviewSurfaceInput.DISPLACEMENT: 0.0,
            PreviewSurfaceInput.OCCLUSION: 1.0,
        }
    )
    """
    Inputs the glTF model cannot express but that are harmless at these values.
    """

    def map_inputs(self, inputs: ShaderInputs) -> PhysicallyBasedMaterial:
        """
        The glTF material for the inputs; the name is filled in by :meth:`map`.

        :param inputs: The authored inputs.
        """
        input_name = PreviewSurfaceInput
        diffuse_texture = inputs.texture(input_name.DIFFUSE_COLOR)
        emissive_texture = inputs.texture(input_name.EMISSIVE_COLOR)
        opacity = inputs.number(input_name.OPACITY, 1.0)
        threshold = inputs.number(input_name.OPACITY_THRESHOLD, 0.0)
        alpha_mode = AlphaMode.OPAQUE
        if threshold > 0.0:
            alpha_mode = AlphaMode.MASK
        elif opacity < 1.0:
            alpha_mode = AlphaMode.BLEND
        return RoughnessMetallicPacking(
            ScalarChannel(
                inputs.number(input_name.ROUGHNESS, 0.5),
                inputs.texture(input_name.ROUGHNESS),
            ),
            ScalarChannel(
                inputs.number(input_name.METALLIC, 0.0),
                inputs.texture(input_name.METALLIC),
            ),
        ).apply(
            PhysicallyBasedMaterial(
                name="",
                base_color_factor=(
                    *(
                        (1.0, 1.0, 1.0)
                        if diffuse_texture is not None
                        else inputs.color(input_name.DIFFUSE_COLOR, self.diffuse_color)
                    ),
                    opacity,
                ),
                base_color_texture=diffuse_texture,
                emissive_factor=(
                    (1.0, 1.0, 1.0)
                    if emissive_texture is not None
                    else inputs.color(input_name.EMISSIVE_COLOR, (0.0, 0.0, 0.0))
                ),
                emissive_texture=emissive_texture,
                normal_texture=inputs.texture(input_name.NORMAL),
                index_of_refraction=inputs.number(input_name.INDEX_OF_REFRACTION, 1.5),
                clearcoat=inputs.number(input_name.CLEARCOAT, 0.0),
                clearcoat_roughness=inputs.number(input_name.CLEARCOAT_ROUGHNESS, 0.01),
                alpha_mode=alpha_mode,
                alpha_cutoff=threshold if threshold > 0.0 else 0.5,
            )
        )


class DisplayColorInput(StrEnum):
    """
    The geometry primvars that shade a prim without a bound material.
    """

    COLOR = "displayColor"
    """
    The colour primvar.
    """

    OPACITY = "displayOpacity"
    """
    The opacity primvar.
    """


@dataclass
class DisplayColorMapping(ShaderMapping):
    """
    Geometry without a material: its display colour, or USD's fallback grey.
    """

    fallback_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """
    The colour Hydra draws geometry with that has neither material nor display colour.
    """

    def map_inputs(self, inputs: ShaderInputs) -> PhysicallyBasedMaterial:
        """
        The glTF material for the inputs; the name is filled in by :meth:`map`.

        :param inputs: The authored inputs.
        """
        opacity = inputs.number(DisplayColorInput.OPACITY, 1.0)
        return PhysicallyBasedMaterial(
            name="",
            base_color_factor=(
                *inputs.color(DisplayColorInput.COLOR, self.fallback_color),
                opacity,
            ),
            alpha_mode=AlphaMode.BLEND if opacity < 1.0 else AlphaMode.OPAQUE,
        )


@dataclass
class UnknownShaderMapping(ShaderMapping):
    """
    A shader no mapping understands: drawn in a colour nobody would author, so it is
    noticed, with every input reported.
    """

    warning_color: tuple[float, float, float] = (1.0, 0.0, 1.0)
    """
    Magenta.
    """

    def map_inputs(self, inputs: ShaderInputs) -> PhysicallyBasedMaterial:
        """
        The warning material, whatever the inputs.

        :param inputs: The authored inputs.
        """
        return PhysicallyBasedMaterial(
            name="", base_color_factor=(*self.warning_color, 1.0), roughness_factor=0.5
        )

    def map(self, material: ImportedMaterial) -> PhysicallyBasedMaterial:
        """
        The warning material, listing the shader's identifier first among the
        unmapped inputs.

        :param material: The authored material.
        """
        mapped = super().map(material)
        return replace(
            mapped,
            unmapped_inputs=(material.shader_identifier, *mapped.unmapped_inputs),
        )


# %% dispatch


@dataclass
class MaterialMapper:
    """
    Picks the mapping for each authored material's shading model.
    """

    mappings: dict[MaterialSource, ShaderMapping] = field(
        default_factory=lambda: {
            MaterialSource.OMNI_PBR: OmniPhysicallyBasedMapping(),
            MaterialSource.OMNI_SURFACE: OmniSurfaceMapping(),
            MaterialSource.OMNI_GLASS: OmniGlassMapping(),
            MaterialSource.PREVIEW_SURFACE: PreviewSurfaceMapping(),
            MaterialSource.DISPLAY_COLOR: DisplayColorMapping(),
            MaterialSource.UNKNOWN: UnknownShaderMapping(),
        }
    )
    """
    The mapping per shading model.
    """

    def map(self, material: ImportedMaterial) -> PhysicallyBasedMaterial:
        """
        The glTF material for an authored material.

        :param material: The authored material.
        """
        return self.mappings[material.source].map(material)
