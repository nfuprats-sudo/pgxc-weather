# pgxc-weather

Wind, rain and thermal map overlays plus point forecast tiles for [PGXC](https://pgxc.app), built from Météo-France's
open AROME (0.025°, 0–51 h) and ARPEGE (0.1°, to 102 h) forecasts and published to
GitHub Pages every 6 hours.

- `build.py` — picks the newest complete run, fetches only the needed GRIB messages
  via HTTP range requests, composites ARPEGE under AROME, and writes `site/`
- `site/manifest.json` — which hours / levels exist, grid bounds, run times
- `site/wind/<YYYYMMDDHH>_<level>.png` — wind speed (Meteo-Parapente palette),
  level = metres above ground (10 = surface); rows resampled to Web Mercator
- `site/wind/<YYYYMMDDHH>_<level>.bin` — int8 u,v (0.5 m/s units) on a 0.2° grid, for arrows
- `site/rain/<YYYYMMDDHH>.png` — rain in the hour ending at that time (mm)
- `forecast.py` — PGXC's own thermal forecast (the app's "PGXC" weather model):
  thermal ceiling = parcel on virtual potential temperature from 60 m above
  ground with +1.0 K, w* = Deardorff from the sensible heat flux, Allen 2006
  updraft profile in the app. ARPEGE everywhere, AROME on top where it has data
- `site/thermal/<YYYYMMDDHH>.png` — thermal ceiling (m AMSL), transparent without thermals
- `site/wind/<YYYYMMDDHH>_<3500..5000>.png/.bin` — wind above 3000 m AGL, from pressure levels
- `site/fc/<YYYYMMDD>/<row>_<col>.bin.gz` — 1°x1° point forecast tiles for the
  windgram (layout in `forecast.write_tiles`); `manifest.json` → `forecast`

Data © Météo-France, [Licence Ouverte / Etalab 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).
