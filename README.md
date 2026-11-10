# Transport US Parking

A lot-management app for the Transport US, LLC RV storage lot at 21825 County Road 38, Goshen, IN.
Drivers tap a stall on a 3D map of the lot, choose the unit type (fifth wheel, travel trailer, toy hauler,
park model, camper van, motorhome), add details and photos, and save. Open stalls are green, occupied stalls are red.

## What's here

| Path | What it is |
|---|---|
| `app/index.html` | The built app: one self-contained page (three.js viewer + the lot model embedded). |
| `model/transport_us_lot.glb` | The 3D model of the lot. X = east, Y = up, Z = -north, 1 unit = 1 m. Each stall is a node named `slot_<id>`. |
| `model/transport_us_lot.json` | Lot data: 351 stalls with IDs and GPS coordinates, rows, drive aisles, zones, buildings, surfaces. |
| `build/build.py` | Generates the GLB, the JSON and `app/index.html` from `build/stalls.json` and `build/template.html`. |
| `build/layout.py` | Generates the organized stall layout (`build/stalls.json`) from the units detected in satellite imagery. |
| `build/detected_units.json` | The 174 parked units detected in the imagery, used to fit the row lines. |
| `analysis/` | The scripts used to stitch the imagery and detect units. Reference only: the imagery itself is not included. |

## Rebuilding

```
pip install trimesh shapely mapbox_earcut numpy
python build/layout.py   # only if the row layout changes
python build/build.py
```

## How the app stores data

The page was built to run as a Claude artifact. There it uses the artifact's shared database for stall records
(collection `stalls`, one document per occupied stall) and its asset store for photos, so everyone it is shared
with sees the same lot. Opened anywhere else, `app/index.html` still works but saves to the browser's local
storage on that device only, without photos. Hosting it as a standalone shared app means replacing the three
functions that touch storage in `build/template.html` (`boot`, `writeStall`, and the photo upload handler)
with your own backend.

## Accuracy

The lot outline, buildings and row lines were measured from Google Maps satellite imagery and are good to
about 1-1.5 m. Building heights are estimates. The lot has no painted stalls: the layout is a proposal that
extends the lines units park on today to the full length of the lot at 4.0 m spacing, with drive aisles and
cross aisles left open.
