# defense — doctoral defense deck

The defense presentation as a local web deck. Slides are plain HTML/CSS/JS;
the *recorded episodes* slide and the title background replay real cramera
scene bundles (full URDF meshes, textures and the recorded giskardpy
trajectory) using the cramera viewer's rendering stack (`vendor/` is copied
from `cramera/src/cramera/web/vendor` on the `cram-viz-integration` branch).

## Run

The deck needs the scene bundles from
[cram2/cram-scenes](https://github.com/cram2/cram-scenes) under
`defense/scenes`:

```bash
cd defense
git clone https://github.com/cram2/cram-scenes scenes   # or: ln -s ~/.cramera/scenes scenes
python3 -m http.server 8123
```

Open <http://localhost:8123>.

## Keys

- `←` / `→` / `Space` — navigate slides
- `t` — light/dark theme
- `f` — fullscreen

On the *recorded episodes* slide: drag to orbit, scroll to zoom, the PR2/HSR/
TIAGO buttons switch between the three recorded apartment episodes, the chips
jump to plan steps.
