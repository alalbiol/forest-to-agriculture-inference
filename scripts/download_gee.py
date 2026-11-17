"""Download annual Sentinel-1/2 composites for one location from Google Earth Engine.

    export GEE_PROJECT=your-earth-engine-project
    python scripts/download_gee.py --lon -50.8762 --lat -3.5342 --years 2020 2024 \
        --chip-id my_chip --output-dir downloads

Reproduces the study's compositing: for each calendar year, the median of Sentinel-2
surface reflectance (COPERNICUS/S2_SR_HARMONIZED, scenes < 70% cloudy, QA60 cloud and
cirrus masked) and of Sentinel-1 IW GRD VV/VH (dB), over a square box of --metres
(440 m in the study) centred on the point, sampled on an explicit 44 x 44 grid. Pixels
with no valid observation are set to 0 and counted in `qa`. Writes
<chip-id>_<year>.npz with `imagery` (14, N, N) and `qa` (2, N, N), ready for inference.py.
"""
import argparse
import io
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import requests

S2_ASSET = 'COPERNICUS/S2_SR_HARMONIZED'
S1_ASSET = 'COPERNICUS/S1_GRD'
S2_CLOUD_LIMIT = 70
S1_BANDS = ['VV', 'VH']
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
QA_BANDS = ['S1_N', 'S2_N']


def bbox_for_point(lon, lat, metres):
    dlat = metres / 2 / 111320.0
    dlon = metres / 2 / (111320.0 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def composite(ee, geometry, year):
    def mask_clouds(image):
        qa = image.select('QA60')
        return image.updateMask(qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0)))

    def blank(bands):
        return ee.Image.constant([0] * len(bands)).rename(bands).toFloat().updateMask(ee.Image(0))

    start, end = f'{year}-01-01', f'{year + 1}-01-01'
    s2 = (ee.ImageCollection(S2_ASSET).filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', S2_CLOUD_LIMIT))
          .map(mask_clouds).select(S2_BANDS).map(lambda i: i.toFloat()))
    s1 = (ee.ImageCollection(S1_ASSET).filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .select(S1_BANDS).map(lambda i: i.toFloat()))
    # An all-masked image keeps the band list valid when a year has no scenes.
    s2 = s2.merge(ee.ImageCollection([blank(S2_BANDS)]))
    s1 = s1.merge(ee.ImageCollection([blank(S1_BANDS)]))
    counts = (s1.select('VV').count().rename('S1_N')
              .addBands(s2.select('B1').count().rename('S2_N')))
    return (s1.median().toFloat().addBands(s2.median().toFloat())
            .addBands(counts.toFloat()).unmask(0).clip(geometry))


def download(image, geometry, cells, attempts=5):
    bands = S1_BANDS + S2_BANDS + QA_BANDS
    for attempt in range(attempts):
        try:
            url = image.getDownloadURL({'region': geometry, 'dimensions': [cells, cells],
                                        'format': 'NPY', 'bands': bands})
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            structured = np.load(io.BytesIO(response.content))
            return np.stack([structured[b].astype(np.float32) for b in bands])
        except Exception:  # transient quota or network errors
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 5) * (0.5 + random.random()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--years', type=int, nargs='+', default=[2020, 2024])
    parser.add_argument('--chip-id', required=True)
    parser.add_argument('--metres', type=int, default=440, help='box side; 440 in the study')
    parser.add_argument('--output-dir', type=Path, default=Path('downloads'))
    parser.add_argument('--project', default=os.getenv('GEE_PROJECT') or os.getenv('EE_PROJECT'))
    args = parser.parse_args()
    if not args.project:
        parser.error('set GEE_PROJECT (or pass --project) to an Earth Engine-enabled Cloud project')
    import ee
    ee.Initialize(project=args.project)
    geometry = ee.Geometry.Rectangle(bbox_for_point(args.lon, args.lat, args.metres))
    cells = args.metres // 10
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        stack = download(composite(ee, geometry, year), geometry, cells)
        if stack.shape != (16, cells, cells) or not np.isfinite(stack).all():
            raise ValueError(f'unexpected download for {year}: {stack.shape}')
        target = args.output_dir / f'{args.chip_id}_{year}.npz'
        np.savez_compressed(target, imagery=stack[:14], qa=stack[14:])
        gaps = [(stack[14 + i] == 0).mean() for i in range(2)]
        print(f'{target}: S1 gaps {gaps[0]:.0%}, S2 gaps {gaps[1]:.0%}')


if __name__ == '__main__':
    main()
