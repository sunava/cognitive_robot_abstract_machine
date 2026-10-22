"""
Mapping of authored USD shaders onto glTF metallic-roughness materials.
"""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy
import pytest
from PIL import Image
from typing_extensions import Any

from cramera.onboard.usd_scene.materials import (
    AlphaMode,
    DisplayColorMapping,
    EmissionScale,
    MaterialMapper,
    OmniGlassInput,
    OmniPhysicallyBasedInput,
    OmniPhysicallyBasedMapping,
    OmniSurfaceInput,
    OmniSurfaceMapping,
    PhysicallyBasedMaterial,
    PreviewSurfaceInput,
    RoughnessMetallicPacking,
    RoughnessMetallicTexture,
    ScalarChannel,
    SrgbTransfer,
    TextureTransform,
    TransmissionBlending,
    UnknownShaderMapping,
)
from cramera.onboard.usd_scene.scene_description import (
    ColorSpace,
    ImportedMaterial,
    MaterialSource,
    TextureChannel,
    TextureFile,
)

TEXTURES = Path(__file__).parent / "dataset" / "usd_scene" / "textures"
"""
Tiny greyscale and colour images whose pixel values the tests compare against.
"""


def texture(name: str, color_space: ColorSpace = ColorSpace.RAW) -> TextureFile:
    """
    A fixture texture.

    :param name: The file name in the texture fixtures.
    :param color_space: The declared colour space.
    """
    return TextureFile(TEXTURES / name, color_space)


def pixels(name: str) -> numpy.ndarray:
    """
    The pixel values of a fixture texture.

    :param name: The file name in the texture fixtures.
    """
    return numpy.asarray(Image.open(TEXTURES / name))


# %% OmniPBR


class TestOmniPhysicallyBasedMapping:
    """
    OmniPBR's albedo, roughness, metalness, emission and tiling.
    """

    def test_untextured_material_uses_omnipbr_defaults(self):
        mapping = OmniPhysicallyBasedMapping()
        material = mapping.map(
            ImportedMaterial("/Looks/Default", MaterialSource.OMNI_PBR)
        )
        assert material.base_color_factor == (*mapping.diffuse_color, 1.0)
        assert material.roughness_factor == mapping.roughness
        assert material.metallic_factor == 0.0
        assert material.metallic_roughness_texture is None

    def test_diffuse_texture_replaces_the_constant_and_keeps_the_tint(self):
        tint = (0.5, 1.0, 1.0)
        albedo = texture("albedo.png", ColorSpace.SRGB)
        material = OmniPhysicallyBasedMapping().map(
            ImportedMaterial(
                "/Looks/Decal",
                MaterialSource.OMNI_PBR,
                {
                    OmniPhysicallyBasedInput.DIFFUSE_TEXTURE: albedo,
                    OmniPhysicallyBasedInput.DIFFUSE_TINT: tint,
                    OmniPhysicallyBasedInput.DIFFUSE_COLOR: (0.9, 0.1, 0.1),
                },
            )
        )
        assert material.base_color_texture == albedo
        assert material.base_color_factor == (*tint, 1.0)

    def test_textured_roughness_and_metalness_share_one_packed_texture(self):
        roughness = texture("roughness.png")
        metallic = texture("metallic.png")
        material = OmniPhysicallyBasedMapping().map(
            ImportedMaterial(
                "/Looks/Steel",
                MaterialSource.OMNI_PBR,
                {
                    OmniPhysicallyBasedInput.ROUGHNESS_TEXTURE: roughness,
                    OmniPhysicallyBasedInput.ROUGHNESS_INFLUENCE: 1.0,
                    OmniPhysicallyBasedInput.METALLIC_TEXTURE: metallic,
                    OmniPhysicallyBasedInput.METALLIC_INFLUENCE: 1.0,
                },
            )
        )
        assert material.metallic_roughness_texture == RoughnessMetallicTexture(
            ScalarChannel(0.5, roughness, 1.0), ScalarChannel(0.0, metallic, 1.0)
        )
        assert (material.roughness_factor, material.metallic_factor) == (1.0, 1.0)

    def test_roughness_texture_without_influence_keeps_the_constant(self):
        material = OmniPhysicallyBasedMapping().map(
            ImportedMaterial(
                "/Looks/Matte",
                MaterialSource.OMNI_PBR,
                {
                    OmniPhysicallyBasedInput.ROUGHNESS: 0.7,
                    OmniPhysicallyBasedInput.ROUGHNESS_TEXTURE: texture(
                        "roughness.png"
                    ),
                },
            )
        )
        assert material.metallic_roughness_texture is None
        assert material.roughness_factor == 0.7

    def test_emission_above_the_reference_moves_into_the_strength(self):
        mapping = OmniPhysicallyBasedMapping()
        intensity = 3.0 * mapping.emission.reference_intensity
        material = mapping.map(
            ImportedMaterial(
                "/Looks/Screen",
                MaterialSource.OMNI_PBR,
                {
                    OmniPhysicallyBasedInput.ENABLE_EMISSION: True,
                    OmniPhysicallyBasedInput.EMISSIVE_COLOR: (1.0, 0.5, 0.0),
                    OmniPhysicallyBasedInput.EMISSIVE_INTENSITY: intensity,
                },
            )
        )
        assert material.emissive_factor == (1.0, 0.5, 0.0)
        assert material.emissive_strength == 3.0

    def test_disabled_emission_emits_nothing(self):
        material = OmniPhysicallyBasedMapping().map(
            ImportedMaterial(
                "/Looks/Off",
                MaterialSource.OMNI_PBR,
                {OmniPhysicallyBasedInput.EMISSIVE_INTENSITY: 4000.0},
            )
        )
        assert material.emissive_factor == (0.0, 0.0, 0.0)

    def test_tiling_becomes_a_texture_transform_in_flipped_coordinates(self):
        material = OmniPhysicallyBasedMapping().map(
            ImportedMaterial(
                "/Looks/Tiles",
                MaterialSource.OMNI_PBR,
                {
                    OmniPhysicallyBasedInput.TEXTURE_SCALE: (4.0, 2.0),
                    OmniPhysicallyBasedInput.TEXTURE_TRANSLATE: (0.25, 0.5),
                },
            )
        )
        assert material.texture_transform == TextureTransform(
            offset=(0.25, 1.0 - 2.0 - 0.5), scale=(4.0, 2.0)
        )

    def test_unknown_authored_inputs_are_listed(self):
        material = OmniPhysicallyBasedMapping().map(
            ImportedMaterial(
                "/Looks/Bright",
                MaterialSource.OMNI_PBR,
                {
                    OmniPhysicallyBasedInput.ALBEDO_BRIGHTNESS: 1.4,
                    OmniPhysicallyBasedInput.SPECULAR_LEVEL: 0.5,
                },
            )
        )
        assert material.unmapped_inputs == (OmniPhysicallyBasedInput.ALBEDO_BRIGHTNESS,)


# %% packed texture


class TestRoughnessMetallicTexture:
    """
    Greyscale roughness and metalness images packed into glTF's green and blue channels.
    """

    def test_green_holds_roughness_and_blue_holds_metalness(self):
        packed = RoughnessMetallicTexture(
            ScalarChannel(0.5, texture("roughness.png")),
            ScalarChannel(0.0, texture("metallic.png")),
        )
        image = numpy.asarray(Image.open(io.BytesIO(packed.encode())))
        assert numpy.array_equal(image[..., 1], pixels("roughness.png"))
        assert numpy.array_equal(image[..., 2], pixels("metallic.png"))

    def test_influence_blends_the_texture_with_the_constant(self):
        packed = RoughnessMetallicTexture(
            ScalarChannel(1.0, texture("roughness.png"), 0.25),
            ScalarChannel(0.0),
        )
        image = numpy.asarray(Image.open(io.BytesIO(packed.encode())))
        expected = numpy.round(0.75 * 255.0 + 0.25 * pixels("roughness.png"))
        assert numpy.array_equal(image[..., 1], expected)
        assert numpy.array_equal(image[..., 2], numpy.zeros_like(expected))

    def test_a_single_channel_of_a_colour_image_is_read(self):
        channel = ScalarChannel(
            0.0, TextureFile(TEXTURES / "albedo.png", channel=TextureChannel.GREEN)
        )
        values = channel.texel_values(None)
        assert numpy.array_equal(values * 255.0, pixels("albedo.png")[..., 1])


# %% transmission drawn by blending


def hemispherical_reflectance(index_of_refraction: float) -> float:
    """
    The closed form of the Fresnel reflectance of unpolarized light, averaged over the
    cosine-weighted hemisphere of directions of incidence from air.

    :param index_of_refraction: The index of refraction of the material.
    """
    index = index_of_refraction
    return (
        0.5
        + (index - 1) * (3 * index + 1) / (6 * (index + 1) ** 2)
        + index**2
        * (index**2 - 1) ** 2
        / (index**2 + 1) ** 3
        * numpy.log((index - 1) / (index + 1))
        - 2 * index**3 * (index**2 + 2 * index - 1) / ((index**2 + 1) * (index**4 - 1))
        + 8
        * index**4
        * (index**4 + 1)
        / ((index**2 + 1) * (index**4 - 1) ** 2)
        * numpy.log(index)
    )


def transmissive(**properties: Any) -> PhysicallyBasedMaterial:
    """
    A fully transmissive, smooth, colourless surface with an index of refraction of one,
    changed by the given properties.

    :param properties: The material properties that differ.
    """
    return replace(
        PhysicallyBasedMaterial(
            "/Looks/Glass",
            roughness_factor=0.0,
            transmission=1.0,
            index_of_refraction=1.0,
        ),
        **properties,
    )


def smoked_tint() -> tuple[float, float, float]:
    """
    The average linear colour of the smoked fixture texture, decoded from sRGB.
    """
    decoded = SrgbTransfer().decoded(pixels("smoke.png") / 255.0)
    return tuple(float(value) for value in decoded.mean(axis=(0, 1)))


class TestTransmissionBlending:
    """
    Transmissive surfaces drawn as alpha-blended, reflecting approximations.
    """

    @pytest.mark.parametrize("index_of_refraction", [1.33, 1.5, 1.52, 1.59, 2.4])
    def test_reflectance_is_the_fresnel_reflectance_averaged_over_the_hemisphere(
        self, index_of_refraction
    ):
        assert TransmissionBlending.reflectance(index_of_refraction) == pytest.approx(
            hemispherical_reflectance(index_of_refraction)
        )

    def test_clear_colourless_glass_covers_the_light_it_reflects(self):
        blended = TransmissionBlending().approximate(
            transmissive(index_of_refraction=1.5)
        )
        assert blended.base_color_factor[3] == pytest.approx(
            hemispherical_reflectance(1.5)
        )

    def test_tint_covers_what_its_least_transmitted_channel_absorbs(self):
        blended = TransmissionBlending().approximate(
            transmissive(base_color_factor=(0.5, 0.8, 1.0, 1.0))
        )
        assert blended.base_color_factor[3] == pytest.approx(0.5)

    def test_frosting_scatters_its_share_out_of_the_line_of_sight(self):
        blended = TransmissionBlending(frosted_scattering=0.5).approximate(
            transmissive(roughness_factor=0.4)
        )
        assert blended.base_color_factor[3] == pytest.approx(0.2)

    def test_light_the_surface_does_not_transmit_is_covered(self):
        blended = TransmissionBlending().approximate(transmissive(transmission=0.25))
        assert blended.base_color_factor[3] == pytest.approx(0.75)

    def test_cut_out_coverage_scales_the_opacity(self):
        blending = TransmissionBlending()
        whole = blending.approximate(transmissive(index_of_refraction=1.5))
        cut = blending.approximate(
            transmissive(
                index_of_refraction=1.5, base_color_factor=(1.0, 1.0, 1.0, 0.5)
            )
        )
        assert cut.base_color_factor[3] == pytest.approx(
            0.5 * whole.base_color_factor[3]
        )

    def test_blend_shows_reflection_and_tinted_transmission_of_even_surroundings(
        self,
    ):
        tint = (0.65, 0.33, 0.06)
        blended = TransmissionBlending().approximate(
            transmissive(index_of_refraction=1.52, base_color_factor=(*tint, 1.0))
        )
        reflectance = TransmissionBlending.reflectance(1.52)
        alpha = blended.base_color_factor[3]
        assert [
            alpha * color + (1.0 - alpha) for color in blended.base_color_factor[:3]
        ] == pytest.approx(
            [reflectance + (1.0 - reflectance) * component for component in tint]
        )

    def test_blended_surface_reflects_without_diffusing(self):
        roughness = texture("roughness.png")
        blended = TransmissionBlending().approximate(
            transmissive(
                metallic_roughness_texture=RoughnessMetallicTexture(
                    ScalarChannel(0.0, roughness), ScalarChannel(0.0)
                )
            )
        )
        assert blended.metallic_factor == 1.0
        assert blended.metallic_roughness_texture == RoughnessMetallicTexture(
            ScalarChannel(0.0, roughness), ScalarChannel(1.0)
        )

    def test_blended_surface_is_drawn_from_both_sides(self):
        blended = TransmissionBlending().approximate(transmissive())
        assert blended.alpha_mode is AlphaMode.BLEND
        assert blended.double_sided

    def test_opaque_surface_is_drawn_as_it_is(self):
        opaque = PhysicallyBasedMaterial("/Looks/Paint")
        assert TransmissionBlending().approximate(opaque) is opaque

    def test_frosting_of_a_packed_constant_roughness_is_that_constant(self):
        packed = RoughnessMetallicPacking(
            ScalarChannel(0.0), ScalarChannel(0.0, texture("metallic.png"))
        ).apply(transmissive(index_of_refraction=1.5))
        blended = TransmissionBlending().approximate(packed)
        assert blended.base_color_factor[3] == pytest.approx(
            hemispherical_reflectance(1.5)
        )

    def test_frosting_of_a_roughness_texture_is_its_average_texel(self):
        packed = RoughnessMetallicPacking(
            ScalarChannel(0.0, texture("roughness.png")), ScalarChannel(0.0)
        ).apply(transmissive())
        blended = TransmissionBlending().approximate(packed)
        assert blended.base_color_factor[3] == pytest.approx(
            pixels("roughness.png").mean() / 255.0
        )

    def test_transmitted_share_of_a_surface_is_metallic(self):
        authored, share = 0.2, 0.1
        blended = TransmissionBlending().approximate(
            transmissive(transmission=share, metallic_factor=authored)
        )
        assert blended.metallic_factor == pytest.approx(
            authored + share * (1.0 - authored)
        )

    def test_metalness_texture_of_a_partly_transmissive_surface_moves_towards_metal(
        self,
    ):
        share = 0.25
        packed = RoughnessMetallicPacking(
            ScalarChannel(0.3), ScalarChannel(0.0, texture("metallic.png"))
        ).apply(transmissive(transmission=share))
        blended = TransmissionBlending().approximate(packed)
        image = numpy.asarray(
            Image.open(io.BytesIO(blended.metallic_roughness_texture.encode()))
        )
        assert numpy.array_equal(
            image[..., 2],
            numpy.round((1.0 - share) * pixels("metallic.png") + share * 255.0),
        )

    def test_tint_texture_covers_like_its_average_colour(self):
        blending = TransmissionBlending()
        textured = blending.approximate(
            transmissive(
                index_of_refraction=1.5,
                base_color_texture=texture("smoke.png", ColorSpace.SRGB),
            )
        )
        flat = blending.approximate(
            transmissive(
                index_of_refraction=1.5, base_color_factor=(*smoked_tint(), 1.0)
            )
        )
        assert textured.base_color_factor[3] == pytest.approx(flat.base_color_factor[3])

    def test_tint_texture_shows_the_blended_colour_at_its_average_texel(self):
        blending = TransmissionBlending()
        textured = blending.approximate(
            transmissive(
                index_of_refraction=1.5,
                base_color_texture=texture("smoke.png", ColorSpace.SRGB),
            )
        )
        flat = blending.approximate(
            transmissive(
                index_of_refraction=1.5, base_color_factor=(*smoked_tint(), 1.0)
            )
        )
        assert [
            factor * tint
            for factor, tint in zip(textured.base_color_factor[:3], smoked_tint())
        ] == pytest.approx(flat.base_color_factor[:3])


def test_srgb_decoding_follows_iec_61966_2_1():
    decoded = SrgbTransfer().decoded(numpy.array([0.0, 0.04045, 0.5, 1.0]))
    assert decoded == pytest.approx([0.0, 0.0031308, 0.2140411, 1.0], abs=1e-7)


# %% OmniSurface, OmniGlass, preview surface


def test_omnisurface_base_colour_is_scaled_by_the_default_diffuse_weight():
    mapping = OmniSurfaceMapping()
    color = (0.6, 0.6, 0.5)
    material = mapping.map(
        ImportedMaterial(
            "/Looks/Steel",
            MaterialSource.OMNI_SURFACE,
            {
                OmniSurfaceInput.DIFFUSE_COLOR: color,
                OmniSurfaceInput.METALNESS: 1.0,
                OmniSurfaceInput.COAT_WEIGHT: 0.5,
            },
        )
    )
    assert material.base_color_factor == (
        *(component * mapping.diffuse_weight for component in color),
        1.0,
    )
    assert material.metallic_factor == 1.0
    assert material.roughness_factor == mapping.roughness
    assert (material.clearcoat, material.clearcoat_roughness) == (
        0.5,
        mapping.coat_roughness,
    )


def test_omnisurface_roughness_image_replaces_the_constant():
    roughness = texture("roughness.png")
    material = OmniSurfaceMapping().map(
        ImportedMaterial(
            "/Looks/Brushed",
            MaterialSource.OMNI_SURFACE,
            {
                OmniSurfaceInput.ROUGHNESS_IMAGE: roughness,
                OmniSurfaceInput.METALNESS: 1.0,
            },
        )
    )
    assert material.metallic_roughness_texture.roughness.texture == roughness
    assert material.metallic_roughness_texture.metallic == ScalarChannel(1.0)


def test_clear_omnisurface_glass_covers_only_the_light_it_reflects():
    material = OmniSurfaceMapping().map(
        ImportedMaterial(
            "/Looks/Glass",
            MaterialSource.OMNI_SURFACE,
            {
                OmniSurfaceInput.ENABLE_TRANSMISSION: True,
                OmniSurfaceInput.TRANSMISSION_WEIGHT: 1.0,
                OmniSurfaceInput.ROUGHNESS: 0.0,
            },
        )
    )
    blended = TransmissionBlending().approximate(material)
    assert blended.base_color_factor[3] == pytest.approx(
        hemispherical_reflectance(material.index_of_refraction)
    )


def test_omnisurface_transmission_colour_tints_the_transmitted_share():
    mapping = OmniSurfaceMapping()
    diffuse, transmitted, share = (0.8, 0.1, 0.1), (0.2, 0.4, 1.0), 0.25
    material = mapping.map(
        ImportedMaterial(
            "/Looks/Tinted",
            MaterialSource.OMNI_SURFACE,
            {
                OmniSurfaceInput.DIFFUSE_COLOR: diffuse,
                OmniSurfaceInput.ENABLE_TRANSMISSION: True,
                OmniSurfaceInput.TRANSMISSION_WEIGHT: share,
                OmniSurfaceInput.TRANSMISSION_COLOR: transmitted,
            },
        )
    )
    assert material.base_color_factor[:3] == pytest.approx(
        [
            (1.0 - share) * mapping.diffuse_weight * diffuse_component
            + share * transmitted_component
            for diffuse_component, transmitted_component in zip(diffuse, transmitted)
        ]
    )


def test_omnisurface_transmission_image_tints_a_mostly_transmissive_surface():
    smoke = texture("smoke.png", ColorSpace.SRGB)
    material = OmniSurfaceMapping().map(
        ImportedMaterial(
            "/Looks/Smoked",
            MaterialSource.OMNI_SURFACE,
            {
                OmniSurfaceInput.DIFFUSE_IMAGE: texture("albedo.png", ColorSpace.SRGB),
                OmniSurfaceInput.ENABLE_TRANSMISSION: True,
                OmniSurfaceInput.TRANSMISSION_WEIGHT: 1.0,
                OmniSurfaceInput.TRANSMISSION_COLOR_IMAGE: smoke,
            },
        )
    )
    assert material.base_color_texture == smoke


def test_omniglass_transmits_blends_and_refracts():
    color = (0.9, 0.95, 1.0)
    material = MaterialMapper().map(
        ImportedMaterial(
            "/Looks/Glass",
            MaterialSource.OMNI_GLASS,
            {
                OmniGlassInput.COLOR: color,
                OmniGlassInput.INDEX_OF_REFRACTION: 1.52,
                OmniGlassInput.FROSTING: 0.1,
            },
        )
    )
    assert material.transmission == 1.0
    assert material.index_of_refraction == 1.52
    assert material.roughness_factor == 0.1
    assert material.base_color_factor == (*color, 1.0)
    assert material.alpha_mode is AlphaMode.BLEND
    assert material.double_sided


def test_omniglass_colour_texture_replaces_the_glass_colour():
    smoke = texture("smoke.png", ColorSpace.SRGB)
    material = MaterialMapper().map(
        ImportedMaterial(
            "/Looks/Smoked",
            MaterialSource.OMNI_GLASS,
            {
                OmniGlassInput.COLOR: (0.9, 0.95, 1.0),
                OmniGlassInput.COLOR_TEXTURE: smoke,
            },
        )
    )
    assert material.base_color_texture == smoke
    assert material.base_color_factor == (1.0, 1.0, 1.0, 1.0)


def test_omniglass_hidden_texture_switch_is_not_reported():
    material = MaterialMapper().map(
        ImportedMaterial(
            "/Looks/Smoked",
            MaterialSource.OMNI_GLASS,
            {
                OmniGlassInput.USE_COLOR_TEXTURE: True,
                OmniGlassInput.COLOR_TEXTURE: texture("smoke.png", ColorSpace.SRGB),
            },
        )
    )
    assert material.unmapped_inputs == ()


def test_preview_surface_opacity_blends_and_texture_replaces_colour():
    albedo = texture("albedo.png", ColorSpace.SRGB)
    material = MaterialMapper().map(
        ImportedMaterial(
            "/Looks/Label",
            MaterialSource.PREVIEW_SURFACE,
            {
                PreviewSurfaceInput.DIFFUSE_COLOR: albedo,
                PreviewSurfaceInput.OPACITY: 0.4,
                PreviewSurfaceInput.METALLIC: 0.2,
            },
        )
    )
    assert material.base_color_texture == albedo
    assert material.base_color_factor == (1.0, 1.0, 1.0, 0.4)
    assert material.alpha_mode is AlphaMode.BLEND
    assert material.metallic_factor == 0.2


def test_unbound_geometry_without_display_colour_is_fallback_grey():
    mapping = DisplayColorMapping()
    material = mapping.map(
        ImportedMaterial("displayColor", MaterialSource.DISPLAY_COLOR)
    )
    assert material.base_color_factor == (*mapping.fallback_color, 1.0)


def test_unknown_shader_is_magenta_and_reported():
    mapping = UnknownShaderMapping()
    material = mapping.map(
        ImportedMaterial(
            "/Looks/Car", MaterialSource.UNKNOWN, {"paint_flake": 0.3}, "CarPaint"
        )
    )
    assert material.base_color_factor == (*mapping.warning_color, 1.0)
    assert material.unmapped_inputs == ("CarPaint", "paint_flake")


@pytest.mark.parametrize("ratio", [0.25, 1.0, 4.0])
def test_emission_scale_preserves_the_emitted_colour(ratio):
    scale = EmissionScale()
    color = (0.5, 0.25, 1.0)
    emission = scale.emission(color, ratio * scale.reference_intensity)
    assert max(emission.factor) <= 1.0
    assert numpy.allclose(
        numpy.array(emission.factor) * emission.strength, numpy.array(color) * ratio
    )
