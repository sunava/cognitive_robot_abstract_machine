"""
Create an offline preview gallery beside the independent instrument packages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path
from string import Template

from catalog import build_devices
from spec import DeviceSpec, JointKind


# %% portable browsing
@dataclass(frozen=True)
class InstrumentGallery:
    """
    Publish previews and local Blender downloads without external services.
    """

    directory: Path
    """
    Catalog root containing the generated instrument folders.
    """

    def write(self, devices: list[DeviceSpec]) -> Path:
        """
        Render a static page with relative links to the complete collection.
        """
        templates = Path(__file__).parent
        card_template = Template((templates / "gallery_card.html").read_text())
        cards = []
        for index, device in enumerate(devices, start=1):
            moving = sum(part.joint != JointKind.FIXED for part in device.parts)
            cards.append(
                card_template.substitute(
                    identifier=escape(device.identifier, quote=True),
                    number=f"{index:02d}",
                    title=escape(device.title, quote=True),
                    description=escape(device.description),
                    joints=moving,
                    joint_label="moving joint" if moving == 1 else "moving joints",
                    frames=len(device.frames),
                    disabled="" if moving else "disabled",
                )
            )
        output = self.directory / "catalog" / "index.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            Template((templates / "gallery.html").read_text()).substitute(
                count=len(devices), cards="\n".join(cards)
            )
        )
        return output


# %% command line
def main() -> None:
    """
    Build the browser overview in an existing generated catalog.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    options = parser.parse_args()
    print(InstrumentGallery(options.root).write(build_devices()))


if __name__ == "__main__":
    main()
