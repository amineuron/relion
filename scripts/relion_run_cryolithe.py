#!/usr/bin/env python3
"""
RELION wrapper for CryoLithe neural network reconstruction.

Integrates as a reconstruction method within the Reconstruct Tomograms job.
Reads an aligned_tilt_series.star, bins tilt series stacks (preserving the
number of tilts in Z), extracts tilt angles, generates a CryoLithe YAML
config, runs super-list.py, and produces output identical to the standard
Reconstruct Tomograms job:
  - tomograms.star (with all original columns + reconstruction columns)
  - tomograms/rec_Position_X.mrc
  - tilt_series/Position_X.star (copied from input)
"""

import os
import sys
import argparse
import subprocess
import shutil
import math


def parse_star_block(star_path, block_name=None):
    """Parse a data block from a RELION STAR file.

    If block_name is given (e.g. 'data_global'), parse that specific block.
    If block_name is None, parse the first data_ block found in the file.

    Returns (header_lines, column_names, data_rows) where each row is a raw string line.
    """
    header_lines = []
    columns = []
    data_lines = []
    in_target_block = False
    in_loop = False

    with open(star_path, 'r') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            stripped = line.strip()

            # Detect start of a data_ block
            if stripped.startswith('data_'):
                if block_name is not None:
                    if stripped == block_name:
                        in_target_block = True
                        header_lines.append(line)
                        continue
                    else:
                        # If we were in a block and hit a new one, stop
                        if in_target_block:
                            break
                        header_lines.append(line)
                        continue
                else:
                    # No specific block requested: use the first one found
                    in_target_block = True
                    header_lines.append(line)
                    continue

            if not in_target_block:
                header_lines.append(line)
                continue

            if stripped == 'loop_':
                in_loop = True
                continue

            if in_loop and stripped.startswith('_'):
                col_parts = stripped.split()
                columns.append(col_parts[0])
                continue

            if in_loop and len(stripped) > 0 and not stripped.startswith('#'):
                data_lines.append(line)
                continue

    return header_lines, columns, data_lines


def parse_star_global(star_path):
    """Parse the data_global block of a RELION STAR file."""
    return parse_star_block(star_path, block_name='data_global')


def parse_row(line, columns):
    """Parse a data row into a dict based on column names."""
    parts = line.split()
    row = {}
    for i, col in enumerate(columns):
        if i < len(parts):
            row[col] = parts[i]
    return row


def write_pipeline_control(output_dir, success=True):
    """Write RELION pipeline control files."""
    fname = 'RELION_JOB_EXIT_SUCCESS' if success else 'RELION_JOB_EXIT_FAILURE'
    with open(os.path.join(output_dir, fname), 'w') as f:
        pass


def fourier_crop_2d(image, new_ny, new_nx):
    """Fourier-crop a single 2D image. Standard cryo-EM downsampling method."""
    import numpy as np
    ny, nx = image.shape
    ft = np.fft.rfft2(image)

    new_nx_ft = new_nx // 2 + 1
    new_ny_half = new_ny // 2

    cropped_ft = np.zeros((new_ny, new_nx_ft), dtype=ft.dtype)

    # Copy positive frequencies (top rows)
    src_top = min(new_ny_half, ny // 2)
    dst_cols = min(new_nx_ft, ft.shape[1])
    cropped_ft[:src_top, :dst_cols] = ft[:src_top, :dst_cols]

    # Copy negative frequencies (bottom rows)
    src_bot = min(new_ny_half, ny // 2)
    cropped_ft[-src_bot:, :dst_cols] = ft[-src_bot:, :dst_cols]

    result = np.fft.irfft2(cropped_ft, s=(new_ny, new_nx))

    # Scale to preserve mean intensity
    scale = (new_ny * new_nx) / (ny * nx)
    result *= scale

    return result.astype(np.float32)


def bin_stack_python(input_path, output_path, bin_factor, target_angpix):
    """Bin a tilt series MRC stack in X,Y only, preserving Z (number of tilts).

    Uses Fourier cropping (standard cryo-EM downsampling) and mrcfile for I/O.
    """
    import mrcfile
    import numpy as np

    with mrcfile.open(input_path, permissive=True) as mrc:
        data = mrc.data.copy()

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    nz, ny, nx = data.shape
    new_nx = int(round(nx / bin_factor))
    new_ny = int(round(ny / bin_factor))

    # Force even dimensions
    if new_nx % 2 != 0:
        new_nx += 1
    if new_ny % 2 != 0:
        new_ny += 1

    print(f"    Stack: {nz} tilts, {nx}x{ny} -> {new_nx}x{new_ny} (Z preserved)")

    binned = np.zeros((nz, new_ny, new_nx), dtype=np.float32)
    for i in range(nz):
        binned[i] = fourier_crop_2d(data[i].astype(np.float32), new_ny, new_nx)

    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(binned)
        mrc.voxel_size = (target_angpix, target_angpix, target_angpix)
        mrc.update_header_stats()

    return nz, new_ny, new_nx


def main():
    parser = argparse.ArgumentParser(description='RELION CryoLithe reconstruction wrapper')
    parser.add_argument('--tiltseries_star', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--cryolithe_script', required=True)
    parser.add_argument('--output_angpix', type=float, required=True)
    parser.add_argument('--x_size', type=int, required=True)
    parser.add_argument('--y_size', type=int, required=True)
    parser.add_argument('--z_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pipeline_control', default='')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tomograms_dir = os.path.join(output_dir, 'tomograms')
    os.makedirs(tomograms_dir, exist_ok=True)

    tilt_series_dir = os.path.join(output_dir, 'tilt_series')
    os.makedirs(tilt_series_dir, exist_ok=True)

    external_dir = os.path.join(output_dir, 'external')
    os.makedirs(external_dir, exist_ok=True)

    binned_dir = os.path.join(external_dir, 'binned_stacks')
    os.makedirs(binned_dir, exist_ok=True)

    tlt_dir = os.path.join(external_dir, 'tlt_files')
    os.makedirs(tlt_dir, exist_ok=True)

    # ---- Read the input STAR file ----
    print(f"Reading input STAR file: {args.tiltseries_star}")
    header_lines, columns, data_lines = parse_star_global(args.tiltseries_star)

    if not columns or not data_lines:
        print("ERROR: could not parse data_global block from input STAR file.")
        write_pipeline_control(output_dir, success=False)
        sys.exit(1)

    rows = [parse_row(line, columns) for line in data_lines]

    proj_files = []
    angle_files = []
    save_names = []
    n3_list = []
    tomo_names = []
    bin_factors = []

    for row in rows:
        tomo_name = row.get('_rlnTomoName', '')
        ts_star_path = row.get('_rlnTomoTiltSeriesStarFile', '')
        ts_angpix = row.get('_rlnTomoTiltSeriesPixelSize', '')
        orig_angpix = row.get('_rlnMicrographOriginalPixelSize', '')

        if not tomo_name or not ts_star_path:
            print(f"WARNING: skipping row without TomoName or TiltSeriesStarFile")
            continue

        angpix = float(ts_angpix) if ts_angpix else float(orig_angpix) if orig_angpix else 1.0
        bin_factor = args.output_angpix / angpix

        print(f"\nProcessing: {tomo_name}")
        print(f"  Pixel size: {angpix:.4f} A -> {args.output_angpix:.4f} A (bin {bin_factor:.2f}x)")

        tomo_names.append(tomo_name)
        bin_factors.append(bin_factor)

        # Find the ALIGNED MRC stack from AlignTiltSeries job.
        # CryoLithe needs the aligned stack (tilt axis rotation + shifts applied).
        # Priority: _aligned.mrc (AreTomo2 or symlink) > _Imod/_st.mrc (AreTomo3)
        # > raw .mrc (last resort, won't have tilt axis correction)
        align_job_dir = os.path.dirname(os.path.dirname(ts_star_path))
        ext_dir = os.path.join(align_job_dir, 'external', tomo_name)

        stack_path = None
        stack_source = 'unknown'
        candidates = [
            (os.path.join(ext_dir, tomo_name + '_aligned.mrc'), 'aligned'),
            (os.path.join(ext_dir, tomo_name + '_Imod', tomo_name + '_st.mrc'), 'Imod_st'),
            (os.path.join(ext_dir, tomo_name + '.mrc'), 'raw'),
        ]
        for cpath, csource in candidates:
            if os.path.exists(cpath):
                stack_path = cpath
                stack_source = csource
                break

        if stack_path is None:
            print(f"  ERROR: no stack found for {tomo_name}")
            print(f"         Searched in: {ext_dir}")
            write_pipeline_control(output_dir, success=False)
            sys.exit(1)

        if stack_source == 'raw':
            print(f"  WARNING: using RAW (unaligned) stack - tilt axis NOT corrected!")
        print(f"  Using stack ({stack_source}): {stack_path}")

        # Bin the stack using Fourier cropping (preserves Z and contrast)
        binned_path = os.path.join(binned_dir, tomo_name + '.mrc')

        if bin_factor > 1.01:
            print(f"  Binning stack (Fourier crop): {stack_path} -> {binned_path}")
            try:
                nz, new_ny, new_nx = bin_stack_python(
                    stack_path, binned_path, bin_factor, args.output_angpix)
                print(f"  Binning complete: {nz} tilts, {new_nx}x{new_ny}")
            except Exception as e:
                print(f"  ERROR binning stack: {e}")
                import traceback
                traceback.print_exc()
                write_pipeline_control(output_dir, success=False)
                sys.exit(1)
            proj_files.append(os.path.abspath(binned_path))
        else:
            print(f"  No binning needed (bin factor {bin_factor:.2f}x)")
            proj_files.append(os.path.abspath(stack_path))

        # Get tilt angles. Priority:
        # 1. _Imod/<name>_st.tlt (matches aligned stack ordering from AreTomo3)
        # 2. Extract from per-tilt-series STAR file (_rlnTomoYTilt)
        imod_tlt = os.path.join(ext_dir, tomo_name + '_Imod', tomo_name + '_st.tlt')
        tlt_path = os.path.join(tlt_dir, tomo_name + '.tlt')

        if os.path.exists(imod_tlt):
            shutil.copy2(imod_tlt, tlt_path)
            n_angles = sum(1 for line in open(tlt_path) if line.strip())
            print(f"  Copied {n_angles} tilt angles from {imod_tlt}")
        else:
            # Extract from per-tilt-series STAR file
            _, ts_cols, ts_data = parse_star_block(ts_star_path, block_name=None)
            ytilt_idx = None
            nom_idx = None
            for ci, cn in enumerate(ts_cols):
                if cn == '_rlnTomoYTilt':
                    ytilt_idx = ci
                if cn == '_rlnTomoNominalStageTiltAngle':
                    nom_idx = ci

            angle_col_idx = ytilt_idx if ytilt_idx is not None else nom_idx
            angle_source = '_rlnTomoYTilt' if ytilt_idx is not None else '_rlnTomoNominalStageTiltAngle'

            n_angles = 0
            with open(tlt_path, 'w') as f:
                if angle_col_idx is not None:
                    for dline in ts_data:
                        parts = dline.split()
                        if angle_col_idx < len(parts):
                            f.write(parts[angle_col_idx] + '\n')
                            n_angles += 1

            if n_angles == 0:
                print(f"  WARNING: No tilt angles extracted from {ts_star_path}")
                print(f"           Columns found: {ts_cols}")
            else:
                print(f"  Wrote {n_angles} tilt angles from STAR ({angle_source})")

        angle_files.append(os.path.abspath(tlt_path))

        # Copy tilt_series STAR file to output (mimic standard reconstruct)
        out_ts_star = os.path.join(tilt_series_dir, tomo_name + '.star')
        shutil.copy2(ts_star_path, out_ts_star)

        save_name = 'rec_' + tomo_name + '.mrc'
        save_names.append(save_name)

        n3 = int(round(args.z_size / bin_factor))
        n3_list.append(n3)
        print(f"  N3 (binned Z): {n3}")

    if not tomo_names:
        print("ERROR: no tomograms found in input STAR file.")
        write_pipeline_control(output_dir, success=False)
        sys.exit(1)

    # ---- Generate CryoLithe YAML config ----
    config_path = os.path.join(output_dir, 'cryolithe_config.yaml')
    tomograms_dir_abs = os.path.abspath(tomograms_dir)

    print(f"\nGenerating CryoLithe config: {config_path}")

    gpu_ids = args.gpu.strip().replace(':', ' ').replace(',', ' ').split()

    with open(config_path, 'w') as f:
        f.write('# CryoLithe config auto-generated by RELION\n')
        f.write(f'model_dir: "{os.path.abspath(args.model_dir)}"\n\n')

        f.write('proj_file:\n')
        for p in proj_files:
            f.write(f'  - "{p}"\n')

        f.write('\nangle_file:\n')
        for a in angle_files:
            f.write(f'  - "{a}"\n')

        f.write(f'\nsave_dir: "{tomograms_dir_abs}"\n')

        f.write('\nsave_name:\n')
        for s in save_names:
            f.write(f'  - "{s}"\n')

        if len(gpu_ids) == 1:
            f.write(f'\ndevice: {gpu_ids[0]}\n')
            f.write('multi_gpu: False\n')
        else:
            f.write(f'\ndevice: [{", ".join(gpu_ids)}]\n')
            f.write('multi_gpu: True\n')

        f.write('\ndownsample_projections: False\n')
        f.write('downsample_factor: 0.25\n')
        f.write('anti_alias: True\n')

        f.write('\nN3:\n')
        for n in n3_list:
            f.write(f'  - {n}\n')

        f.write(f'\nbatch_size: {args.batch_size}\n')
        f.write(f'num_workers: {args.num_workers}\n')

    # ---- Run CryoLithe ----
    cryolithe_cmd = [sys.executable, args.cryolithe_script,
                     '--config', os.path.abspath(config_path)]
    print(f"\nRunning CryoLithe: {' '.join(cryolithe_cmd)}")
    result = subprocess.run(cryolithe_cmd)
    if result.returncode != 0:
        print(f"ERROR: CryoLithe failed with return code {result.returncode}")
        write_pipeline_control(output_dir, success=False)
        sys.exit(1)

    # ---- Fix MRC headers on CryoLithe output tomograms ----
    import mrcfile
    import numpy as np
    print("\nFixing MRC headers on output tomograms...")
    for idx, save_name in enumerate(save_names):
        tomo_path = os.path.join(tomograms_dir, save_name)
        if not os.path.exists(tomo_path):
            print(f"  WARNING: expected output not found: {tomo_path}")
            continue
        with mrcfile.open(tomo_path, mode='r+', permissive=True) as mrc:
            nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
            angpix = args.output_angpix
            # Set grid sampling to match image dimensions
            mrc.header.mx = nx
            mrc.header.my = ny
            mrc.header.mz = nz
            # Set cell dimensions (grid * pixel size)
            mrc.header.cella.x = float(nx) * angpix
            mrc.header.cella.y = float(ny) * angpix
            mrc.header.cella.z = float(nz) * angpix
            # Space group 0 = image stack / volume (not crystal)
            mrc.header.ispg = 0
            # Update min/max/mean statistics
            mrc.update_header_stats()
        print(f"  Fixed: {save_name} ({nx}x{ny}x{nz}, {angpix:.2f} A/px)")

    # ---- Write output tomograms.star (mimic standard reconstruct format) ----
    output_star = os.path.join(output_dir, 'tomograms.star')
    print(f"\nWriting output STAR file: {output_star}")

    out_columns = list(columns)
    extra_cols = [
        '_rlnTomoTomogramBinning',
        '_rlnTomoSizeX',
        '_rlnTomoSizeY',
        '_rlnTomoSizeZ',
        '_rlnTomoReconstructedTomogram',
    ]
    for ec in extra_cols:
        if ec not in out_columns:
            out_columns.append(ec)

    with open(output_star, 'w') as f:
        f.write('\n# version 50001\n\ndata_global\n\nloop_ \n')
        for ci, col in enumerate(out_columns):
            f.write(f'{col} #{ci+1} \n')

        for i, row in enumerate(rows):
            tomo_name = row.get('_rlnTomoName', '')
            if tomo_name not in tomo_names:
                continue

            idx = tomo_names.index(tomo_name)
            bin_factor = bin_factors[idx]
            binned_x = int(round(args.x_size / bin_factor))
            binned_y = int(round(args.y_size / bin_factor))

            new_row = dict(row)
            new_row['_rlnTomoTiltSeriesStarFile'] = os.path.join(
                output_dir, 'tilt_series', tomo_name + '.star')
            new_row['_rlnTomoTomogramBinning'] = f'{bin_factor:.6f}'
            new_row['_rlnTomoSizeX'] = str(binned_x)
            new_row['_rlnTomoSizeY'] = str(binned_y)
            new_row['_rlnTomoSizeZ'] = str(n3_list[idx])
            new_row['_rlnTomoReconstructedTomogram'] = os.path.join(
                output_dir, 'tomograms', save_names[idx])

            parts = []
            for col in out_columns:
                parts.append(new_row.get(col, '0'))
            f.write('\t'.join(parts) + ' \n')

        f.write(' \n')

    print(f"Output STAR written: {output_star}")

    # Pipeline control
    pipeline_dir = args.pipeline_control if args.pipeline_control else output_dir
    write_pipeline_control(pipeline_dir, success=True)
    print("\nCryoLithe reconstruction completed successfully!")


if __name__ == '__main__':
    main()
