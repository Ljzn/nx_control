defmodule NxControl.Vision do
  @moduledoc """
  Image "sight" — analyze an image and turn it into a machine-readable report.

  `NxControl.Vision` gives a large language model a way to *read* an image
  without any multimodal weights: it reduces a bitmap to a compact text report
  (color statistics, a coarse color grid, edge/structure layout, the singular
  value spectrum and a per-cell texture map). The report is intentionally
  plain text so it can be fed directly into an LLM prompt.

  Everything is implemented with pure `Nx` (plus `Nx.Lapack` for the SVD) —
  no external image or machine-learning dependencies.

  ## Quick start

      img = NxControl.Vision.read_bmp("examples/images/svd_cat/orange_cat_original.bmp")
      report = NxControl.Vision.analyze(img)
      IO.puts(NxControl.Vision.to_text(report))

  Or analyze a path directly:

      NxControl.Vision.analyze("examples/images/svd_cat/orange_cat_original.bmp")

  ## Sight helpers

      NxControl.Vision.ascii(img)             # color-letter grid (image -> text)
      NxControl.Vision.compress(img, k: 32)   # SVD low-rank reconstruction
      NxControl.Vision.mosaic(img, cells: 12) # block-average "sketch"

  ## Image I/O

      NxControl.Vision.read_bmp(path)         # -> {H, W, 3} :u8 tensor
      NxControl.Vision.write_bmp(path, img)   # 24-bit BMP, no dependencies

  """
  alias NxControl.Vision.Report

  @hue_names [
    {"red", 0},
    {"orange", 1},
    {"yellow", 2},
    {"yellow-green", 3},
    {"green", 4},
    {"cyan", 5},
    {"cyan-blue", 6},
    {"blue", 7},
    {"blue-violet", 8},
    {"violet", 9},
    {"magenta", 10},
    {"red-violet", 11}
  ]

  # ─────────────────────────────────────────────
  # Image I/O (24-bit BMP, no dependencies)
  # ─────────────────────────────────────────────

  @doc """
  Reads a 24-bit BMP file into an `{H, W, 3}` tensor of type `:u8`.

  Raises if the file is not a 24-bit BMP.
  """
  def read_bmp(path) do
    bin = File.read!(path)
    parse_bmp(bin)
  end

  defp parse_bmp(bin) do
    <<0x42, 0x4D, _size::little-32, _::little-32, data_off::little-32, _hdr::little-32,
      w::little-32, h::little-32, _planes::little-16, bpp::little-16, _compression::little-32,
      _rest::binary>> = bin

    if bpp != 24 do
      raise ArgumentError, "expected a 24-bit BMP, got #{bpp}-bit"
    end

    pixels = binary_part(bin, data_off, byte_size(bin) - data_off)
    row_size = w * 3 + rem(4 - rem(w * 3, 4), 4)

    rows =
      for y <- 0..(h - 1) do
        # BMP rows are stored bottom-up
        src = (h - 1 - y) * row_size
        row_bin = binary_part(pixels, src, w * 3)
        # BGR -> RGB
        for <<b, g, r <- row_bin>>, into: <<>>, do: <<r, g, b>>
      end

    rows
    |> :erlang.list_to_binary()
    |> then(&Nx.from_binary(&1, :u8))
    |> Nx.reshape({h, w, 3})
  end

  @doc """
  Writes an `{H, W, 3}` tensor (values 0..255) as a 24-bit BMP file.

  Returns the file size in bytes.
  """
  def write_bmp(path, rgb) do
    {h, w, 3} = Nx.shape(rgb)
    bin = Nx.to_binary(Nx.as_type(rgb, :u8))
    row_pad = rem(4 - rem(w * 3, 4), 4)
    row_size = w * 3 + row_pad
    data_size = row_size * h
    file_size = 54 + data_size

    header =
      <<0x42, 0x4D, file_size::little-32, 0::little-32, 54::little-32, 40::little-32,
        w::little-32, h::little-32, 1::little-16, 24::little-16, 0::little-32,
        data_size::little-32, 2835::little-32, 2835::little-32, 0::little-32, 0::little-32>>

    rows =
      for row <- (h - 1)..0//-1 do
        row_bin = binary_part(bin, row * w * 3, w * 3)
        bgr = for <<r, g, b <- row_bin>>, into: <<>>, do: <<b, g, r>>
        bgr <> :binary.copy(<<0>>, row_pad)
      end

    File.write!(path, header <> IO.iodata_to_binary(rows))
    file_size
  end

  # ─────────────────────────────────────────────
  # Analyze
  # ─────────────────────────────────────────────

  @doc """
  Analyzes an image and returns a `NxControl.Vision.Report`.

  `image` may be an `{H, W, 3}` tensor or a path to a 24-bit BMP.

  ## Options

    * `:cells` — grid resolution used for the color grid, edges and texture
      (default `12`).
    * `:max_size` — the analysis works on a downsampled copy whose largest
      dimension is at most `:max_size` (default `64`). Keep it small for
      speed; the color grid is what you read, not the raw pixels.

  ## Example

      report = NxControl.Vision.analyze("cat.bmp")
      IO.puts(NxControl.Vision.to_text(report))
  """
  def analyze(image, opts \\ [])

  def analyze(image, opts) when is_binary(image) do
    image |> read_bmp() |> analyze(opts)
  end

  def analyze(%Nx.Tensor{} = image, opts) do
    cells = opts[:cells] || 12
    max_size = opts[:max_size] || 64

    rgb = ensure_rgb(image)
    {h, w, 3} = Nx.shape(rgb)
    work = maybe_resize(Nx.as_type(rgb, :f64), max_size)

    colors = color_stats(work)
    hues = hue_histogram(work)
    grid = grid(work, cells)
    gray = luminance(work)
    edges = edge_analysis(gray, cells)
    svd = svd_spectrum(gray)
    texture = texture_analysis(gray, opts[:texture_patch] || 8)

    %Report{
      shape: {h, w},
      colors: colors,
      hues: hues,
      grid: grid,
      edges: edges,
      svd: svd,
      texture: texture
    }
  end

  defp ensure_rgb(%{shape: {h, w, 3}} = t), do: t

  defp ensure_rgb(%{shape: {h, w, 1}} = t) do
    Nx.broadcast(t, {h, w, 3})
  end

  defp ensure_rgb(t) do
    raise ArgumentError,
          "expected an {H, W, 3} or {H, W, 1} image tensor, got #{inspect(Nx.shape(t))}"
  end

  # Block-average downsample so the largest dimension is at most `max`.
  defp maybe_resize(img, max) do
    {h, w, 3} = Nx.shape(img)
    if max(h, w) <= max, do: img, else: resize(img, max)
  end

  defp resize(img, max) do
    {h, w, 3} = Nx.shape(img)
    scale = max / max(h, w)
    th = max(1, round(h * scale))
    tw = max(1, round(w * scale))
    step_h = max(1, div(h + th - 1, th))
    step_w = max(1, div(w + tw - 1, tw))
    nh = div(h, step_h)
    nw = div(w, step_w)

    img
    |> Nx.slice([0, 0, 0], [nh * step_h, nw * step_w, 3])
    |> Nx.reshape({nh, step_h, nw, step_w, 3})
    |> Nx.mean(axes: [1, 3])
  end

  # Block-average to an approximately G×G grid of RGB cells.
  defp grid(img, cells) do
    {h, w, 3} = Nx.shape(img)
    step_h = max(1, div(h + cells - 1, cells))
    step_w = max(1, div(w + cells - 1, cells))
    nh = div(h, step_h)
    nw = div(w, step_w)

    img
    |> Nx.slice([0, 0, 0], [nh * step_h, nw * step_w, 3])
    |> Nx.reshape({nh, step_h, nw, step_w, 3})
    |> Nx.mean(axes: [1, 3])
  end

  @doc """
  Returns the grayscale (luminance) image: `0.299 R + 0.587 G + 0.114 B`.
  """
  def luminance(%{shape: {_, _, 3}} = rgb) do
    {r, g, b} = split_channels(Nx.as_type(rgb, :f64))
    Nx.add(Nx.add(Nx.multiply(0.299, r), Nx.multiply(0.587, g)), Nx.multiply(0.114, b))
  end

  defp split_channels(%{shape: {h, w, 3}} = rgb) do
    c = Nx.reshape(Nx.transpose(rgb, axes: [2, 0, 1]), {3, h, w})
    {c[0], c[1], c[2]}
  end

  # ─────────────────────────────────────────────
  # Feature extractors
  # ─────────────────────────────────────────────

  defp color_stats(rgb) do
    {h, w, 3} = Nx.shape(rgb)
    {r, g, b} = split_channels(rgb)

    stats = fn ch ->
      flat = Nx.reshape(ch, {h * w})
      m = Nx.mean(flat)
      sd = Nx.sqrt(Nx.mean(Nx.pow(Nx.subtract(flat, m), 2)))
      mn = Nx.reduce_min(flat)
      mx = Nx.reduce_max(flat)

      %{
        mean: Nx.to_number(m),
        sd: Nx.to_number(sd),
        min: Nx.to_number(mn),
        max: Nx.to_number(mx)
      }
    end

    %{r: stats.(r), g: stats.(g), b: stats.(b), luminance: Nx.to_number(Nx.mean(luminance(rgb)))}
  end

  # RGB -> hue histogram with gray/black/white buckets. Returns the top-3
  # categories as `[{name, fraction}]`.
  defp hue_histogram(rgb) do
    flat = Nx.to_flat_list(Nx.as_type(rgb, :f64))
    n = div(length(flat), 3)

    counts =
      Enum.reduce(Enum.chunk_every(flat, 3), %{}, fn [r, g, b], acc ->
        key = hue_key(r, g, b)
        Map.update(acc, key, 1, &(&1 + 1))
      end)

    counts
    |> Enum.map(fn {key, cnt} -> {key, cnt / n} end)
    |> Enum.sort_by(fn {_, f} -> -f end)
    |> Enum.take(3)
  end

  defp hue_key(r, g, b) do
    mx = max(max(r, g), b)
    mn = min(min(r, g), b)
    delta = mx - mn

    cond do
      mx < 45 ->
        "black"

      mn > 210 ->
        "white"

      delta / mx < 0.15 ->
        "gray"

      mx > 215 and delta / mx < 0.3 ->
        "pale"

      true ->
        hue =
          cond do
            mx == r -> 60 * :math.fmod((g - b) / delta, 6)
            mx == g -> 60 * ((b - r) / delta + 2)
            true -> 60 * ((r - g) / delta + 4)
          end

        bin = rem(trunc(hue / 30), 12)
        Enum.find(@hue_names, fn {_, b} -> b == bin end) |> elem(0)
    end
  end

  # Sobel edges on the grayscale image.
  defp edge_analysis(gray, cells) do
    {h, w} = Nx.shape(gray)
    x = Nx.reshape(gray, {1, 1, h, w})

    kx =
      Nx.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
      |> Nx.reshape({1, 1, 3, 3})

    ky =
      Nx.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
      |> Nx.reshape({1, 1, 3, 3})

    # Valid convolution (no padding) so a constant image has zero gradient;
    # zero-padding would otherwise create spurious border edges.
    gx = Nx.conv(x, kx, strides: [1, 1], padding: [{0, 0}, {0, 0}]) |> Nx.reshape({h - 2, w - 2})
    gy = Nx.conv(x, ky, strides: [1, 1], padding: [{0, 0}, {0, 0}]) |> Nx.reshape({h - 2, w - 2})
    mag = Nx.sqrt(Nx.add(Nx.pow(gx, 2), Nx.pow(gy, 2)))

    gx_list = Nx.to_flat_list(gx)
    gy_list = Nx.to_flat_list(gy)
    mag_list = Nx.to_flat_list(mag)

    # orientation histogram (8 directions, magnitude-weighted)
    bins = List.duplicate(0.0, 8)

    {orientation, total} =
      Enum.zip([gx_list, gy_list, mag_list])
      |> Enum.reduce({bins, 0.0}, fn {gx_v, gy_v, m}, {bins_acc, t} ->
        angle = :math.atan2(gy_v, gx_v)
        bin = rem(round(Float.floor((angle + :math.pi()) / (:math.pi() / 4))), 8)
        {List.update_at(bins_acc, bin, &(&1 + m)), t + m}
      end)

    orientation_pct =
      if total > 0 do
        Enum.map(orientation, &(&1 / total * 100))
      else
        List.duplicate(0.0, 8)
      end

    # per-cell edge density
    cell_mag = block_mean_2d(mag, cells)

    %{
      total: total,
      density: cell_mag,
      orientation: orientation_pct,
      dir_labels: ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    }
  end

  defp block_mean_2d(t, cells) do
    {h, w} = Nx.shape(t)
    step_h = max(1, div(h + cells - 1, cells))
    step_w = max(1, div(w + cells - 1, cells))
    nh = div(h, step_h)
    nw = div(w, step_w)

    t
    |> Nx.slice([0, 0], [nh * step_h, nw * step_w])
    |> Nx.reshape({nh, step_h, nw, step_w})
    |> Nx.mean(axes: [1, 3])
    |> Nx.to_flat_list()
  end

  # Singular value spectrum of the grayscale image.
  defp svd_spectrum(gray) do
    {h, w} = Nx.shape(gray)
    {nil, s, nil} = Nx.Lapack.svd(gray, vectors: :none)
    values = Nx.to_flat_list(s)
    energy = Enum.map(values, &(&1 * &1))
    total = Enum.sum(energy)
    cum = Enum.scan(energy, 0, &(&1 + &2))
    norm = Enum.map(cum, &(&1 / total))

    k_for = fn p ->
      Enum.find_index(norm, &(&1 >= p)) || length(values)
    end

    %{
      size: {h, w},
      top8: Enum.take(values, 8),
      k50: max(1, k_for.(0.5)),
      k80: max(1, k_for.(0.8)),
      k95: max(1, k_for.(0.95)),
      eff_rank: max(1, k_for.(0.95))
    }
  end

  # Per-cell texture: first-singular-value share lambda_1 / sum(lambda) of
  # each patch. Flat cells -> near 1.0; detailed cells -> lower.
  defp texture_analysis(gray, patch) do
    {h, w} = Nx.shape(gray)
    ph = min(patch, h)
    pw = min(patch, w)
    nh = div(h, ph)
    nw = div(w, pw)

    flatness =
      for cy <- 0..(nh - 1), cx <- 0..(nw - 1) do
        p = Nx.slice(gray, [cy * ph, cx * pw], [ph, pw])
        {nil, s, nil} = Nx.Lapack.svd(p, vectors: :none)
        sv = Nx.to_flat_list(s)
        first = hd(sv)
        sum = Enum.sum(sv)
        if sum > 0, do: first / sum, else: 1.0
      end

    # A cell is "detailed" if the leading singular value carries less than
    # 80% of the patch energy.
    detail =
      Enum.with_index(flatness)
      |> Enum.filter(fn {f, _} -> f < 0.8 end)
      |> Enum.map(fn {_, i} -> {div(i, nw), rem(i, nw)} end)

    %{
      patch: ph,
      mean_flatness: Enum.sum(flatness) / max(1, length(flatness)),
      grid: flatness,
      detail_cells: detail
    }
  end

  # ─────────────────────────────────────────────
  # Text report
  # ─────────────────────────────────────────────

  @doc """
  Renders a `NxControl.Vision.Report` as a readable multi-line text summary.
  """
  def to_text(%Report{} = r) do
    {h, w} = r.shape
    c = r.colors
    g = r.grid
    {gh, gw, 3} = Nx.shape(g)

    color_line =
      "R mean=#{fmt(c.r.mean)} sd=#{fmt(c.r.sd)} | " <>
        "G mean=#{fmt(c.g.mean)} sd=#{fmt(c.g.sd)} | " <>
        "B mean=#{fmt(c.b.mean)} sd=#{fmt(c.b.sd)} ; luminance=#{fmt(c.luminance)}"

    hues_line = Enum.map_join(r.hues, " | ", fn {name, f} -> "#{name} #{round(f * 100)}%" end)

    edge = r.edges

    density_counts = %{
      hi: Enum.count(edge.density, &(&1 > 25)),
      med: Enum.count(edge.density, &(&1 >= 10 and &1 <= 25)),
      lo: Enum.count(edge.density, &(&1 < 10))
    }

    dirs =
      Enum.zip(edge.dir_labels, edge.orientation)
      |> Enum.sort_by(fn {_, p} -> -p end)
      |> Enum.take(3)
      |> Enum.map_join(" ", fn {lab, p} -> "#{lab}#{round(p)}%" end)

    svd = r.svd
    {wh, ww} = svd.size

    svd_line =
      "top8=[#{Enum.map_join(svd.top8, " ", &fmt/1)}] ; k50%=#{svd.k50} k80%=#{svd.k80} k95%=#{svd.k95} eff_rank=#{svd.eff_rank} (at #{ww}x#{wh})"

    tex = r.texture
    n_detail = length(tex.detail_cells)

    tex_line =
      "flatness=#{fmt(tex.mean_flatness)} (#{tex.patch}px patches) ; detail_cells=#{n_detail} #{inspect(Enum.take(tex.detail_cells, 8))}"

    grid_text =
      g
      |> Nx.to_flat_list()
      |> Enum.chunk_every(3)
      |> Enum.chunk_every(gw)
      |> Enum.map(fn row ->
        Enum.map_join(row, "", fn [rr, gg, bb] -> color_letter(rr, gg, bb) end)
      end)
      |> Enum.join("\n")

    """
    IMAGE #{w}x#{h}
    COLOR  #{color_line}
    HUES   #{hues_line}
    GRID #{gw}x#{gh}
    #{grid_text}
    EDGE   total=#{fmt(edge.total)} ; cells hi=#{density_counts.hi} med=#{density_counts.med} lo=#{density_counts.lo} ; dirs #{dirs}
    SVD    #{svd_line}
    TEXTURE #{tex_line}
    """
  end

  defp fmt(x), do: Float.round(x, 1) |> Float.to_string()

  @doc """
  Prints an image as a coarse grid of color letters (image -> text).

  Each cell is averaged and mapped to a single character:
  `K` black `W` white `R` red `O` orange `Y` yellow `G` green `C` cyan
  `B` blue `M` magenta `n` brown `c` cream `p` pink `g` gray `.` other.

  Returns the multi-line string.
  """
  def ascii(image, opts \\ []) do
    cells = opts[:cells] || 16
    img = analyze(image, cells: cells, max_size: opts[:max_size] || 128)
    {gh, gw, _} = Nx.shape(img.grid)

    img.grid
    |> Nx.to_flat_list()
    |> Enum.chunk_every(3)
    |> Enum.chunk_every(gw)
    |> Enum.map(fn row -> Enum.map_join(row, "", fn [r, g, b] -> color_letter(r, g, b) end) end)
    |> Enum.join("\n")
  end

  defp color_letter(r, g, b) do
    cond do
      r < 60 and g < 60 and b < 60 -> "K"
      r > 200 and g > 200 and b > 200 -> "W"
      r > 200 and g >= 90 and g <= 165 and b >= 90 and b <= 165 -> "p"
      r > 150 and g < 110 and b < 110 -> "R"
      r > 160 and g >= 110 and g < 200 and b < 130 -> "O"
      r > 150 and g >= 150 and b < 130 -> "Y"
      r < 110 and g > 150 and b < 110 -> "G"
      r < 110 and g > 130 and b > 150 -> "C"
      r < 110 and g < 110 and b > 150 -> "B"
      r > 150 and g < 110 and b > 150 -> "M"
      r >= 100 and r < 190 and g >= 60 and g < 135 and b < 95 -> "n"
      r > 205 and g >= 165 and g <= 220 and b >= 130 and b <= 200 -> "c"
      r > 130 and r < 215 and g > 130 and g < 215 and b > 130 and b < 215 -> "g"
      true -> "."
    end
  end

  # ─────────────────────────────────────────────
  # Reconstruction
  # ─────────────────────────────────────────────

  @doc """
  Reconstructs the image from its rank-`k` SVD approximation (per RGB channel).

  This is the "painting" from the singular value spectrum: low `k` gives a
  smooth, abstract rendering; high `k` approaches the original.

  Returns an `{H, W, 3}` tensor with values in 0..255.
  """
  def compress(image, opts \\ []) do
    k = opts[:k] || 32
    rgb = ensure_rgb(Nx.as_type(image, :f64))
    {h, w, 3} = Nx.shape(rgb)

    channels =
      Enum.map(0..2, fn c ->
        ch = Nx.slice(rgb, [0, 0, c], [h, w, 1]) |> Nx.reshape({h, w})
        {u, s, vt} = Nx.Lapack.svd(ch)
        n = Nx.size(s)
        kk = min(k, n)
        uk = Nx.slice(u, [0, 0], [n, kk])
        sk = Nx.slice(s, [0], [kk])
        vk = Nx.slice(vt, [0, 0], [kk, n])
        sm = Nx.multiply(Nx.eye(kk, type: :f64), Nx.reshape(sk, {kk, 1}))
        Nx.dot(Nx.dot(uk, sm), vk)
      end)

    Nx.stack(channels, axis: 2)
    |> Nx.reshape({h, w, 3})
    |> Nx.clip(0.0, 255.0)
  end

  @doc """
  Returns a block-average "mosaic" sketch of the image, upscaled back to the
  original size. Each of the `:cells` × `:cells` regions is flattened to its
  average color.

  Returns an `{H, W, 3}` tensor with values in 0..255.
  """
  def mosaic(image, opts \\ []) do
    cells = opts[:cells] || 12
    rgb = ensure_rgb(Nx.as_type(image, :f64))
    {h, w, 3} = Nx.shape(rgb)
    small = grid(rgb, cells)
    {gh, gw, _} = Nx.shape(small)
    step_h = max(1, div(h + gh - 1, gh))
    step_w = max(1, div(w + gw - 1, gw))

    small
    |> Nx.reshape({gh, 1, gw, 1, 3})
    |> Nx.broadcast({gh, step_h, gw, step_w, 3})
    |> Nx.reshape({gh * step_h, gw * step_w, 3})
    |> Nx.clip(0.0, 255.0)
  end
end
