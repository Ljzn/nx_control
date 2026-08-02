# SVD-Painted Orange Cat
#
# Draw an orange cat procedurally (no external images), decompose each RGB
# channel with Nx.Lapack.svd, reconstruct at a low rank (the "painting"),
# and save the result as a 24-bit BMP bitmap.
#
# Run:  cd nx_control && mix run examples/svd_cat.exs
#
# The cat is drawn to feel warm and serene: soft golden light, half-closed
# gentle eyes and a faint smile — a calm, cozy mood expressed through
# light and geometry rather than photographic detail.

defmodule OrangeCat do
  # Procedurally render an orange tabby cat into an H×W×3 RGB tensor (0..255).
  def render(h, w) do
    Enum.reduce((h - 1)..0//-1, [], fn y, acc ->
      row =
        Enum.map(0..(w - 1), fn x ->
          {r, g, b} = pixel(x, y, w, h)
          [r, g, b]
        end)

      [row | acc]
    end)
    |> List.flatten()
    |> then(&Nx.tensor(&1, type: :f64))
    |> Nx.reshape({h, w, 3})
  end

  defp pixel(x, y, w, h) do
    nx = (x + 0.5) / w * 100.0
    ny = (y + 0.5) / h * 100.0

    # warm light source (top-left) for soft 3D shading
    dist = :math.sqrt((nx - 26) * (nx - 26) + (ny - 30) * (ny - 30))
    shade = max(0.0, min(1.0, 1.0 - dist / 95.0))

    in_left_ear = in_triangle(nx, ny, [[36, 12], [18, 46], [46, 42]])
    in_right_ear = in_triangle(nx, ny, [[64, 12], [82, 46], [54, 42]])
    in_head = in_ellipse(nx, ny, 50, 58, 26, 25)
    in_body = in_ellipse(nx, ny, 50, 96, 35, 22)

    cond do
      # ears (outer + inner)
      in_left_ear or in_right_ear ->
        ear_color(ny, shade)

      in_head or in_body ->
        fur_color(nx, ny, in_head, shade)

      true ->
        background(nx, ny)
    end
  end

  defp in_ellipse(x, y, cx, cy, rx, ry) do
    ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
  end

  defp in_triangle(px, py, [[ax, ay], [bx, by], [cx, cy]]) do
    d1 = sign(px, py, ax, ay, bx, by)
    d2 = sign(px, py, bx, by, cx, cy)
    d3 = sign(px, py, cx, cy, ax, ay)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    not (has_neg and has_pos)
  end

  defp sign(p1x, p1y, p2x, p2y, p3x, p3y) do
    (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
  end

  # Warm orange fur with darker tabby stripes and radial lighting.
  defp fur_color(nx, ny, in_head, shade) do
    # base coat
    base_r = 250.0
    base_g = 158.0
    base_b = 62.0

    # tabby stripes: darker bands
    stripe =
      cond do
        in_head and rem(floor(nx / 6), 2) == 0 and abs(ny - 52) < 14 -> 0.75
        not in_head and rem(floor((nx + ny) / 7), 2) == 0 -> 0.7
        true -> 1.0
      end

    light = 0.45 + 0.55 * shade
    r = base_r * stripe * light
    g = base_g * stripe * light
    b = base_b * stripe * light

    # face features layered on top
    add_face(nx, ny, r, g, b)
  end

  defp add_face(nx, ny, r, g, b) do
    eye_l = in_ellipse(nx, ny, 40, 52, 4.2, 3.0)
    eye_r = in_ellipse(nx, ny, 60, 52, 4.2, 3.0)
    eye_l_top = in_ellipse(nx, ny, 40, 50.5, 2.0, 1.2)
    eye_r_top = in_ellipse(nx, ny, 60, 50.5, 2.0, 1.2)
    nose = in_triangle(nx, ny, [[50, 58], [46.5, 61], [53.5, 61]])
    smile_l = in_ellipse(nx, ny, 47.5, 62.5, 2.2, 1.1)
    smile_r = in_ellipse(nx, ny, 52.5, 62.5, 2.2, 1.1)

    cond do
      # half-closed gentle eyes (upper half of the almond)
      (eye_l or eye_r) and ny <= 52.5 ->
        {38, 30, 26}

      # soft warm eyelid highlights
      eye_l_top or eye_r_top ->
        {210, 150, 90}

      # pink nose
      nose ->
        {222, 120, 108}

      # faint smile (two short arcs)
      smile_l or smile_r ->
        {120, 62, 40}

      # whiskers
      whisker(nx, ny) ->
        {225, 190, 150}

      true ->
        {r, g, b}
    end
  end

  defp whisker(nx, ny) do
    # three thin whiskers per side
    Enum.any?([{62, 56}, {63, 59}, {62, 62}], fn {wx, wy} ->
      abs(ny - wy) < 0.8 and nx >= wx and nx <= wx + 12
    end) or
      Enum.any?([{38, 56}, {37, 59}, {38, 62}], fn {wx, wy} ->
        abs(ny - wy) < 0.8 and nx <= wx and nx >= wx - 12
      end)
  end

  defp ear_color(ny, shade) do
    # inner pink at the ear base, orange otherwise
    if ny > 38 do
      {226, 124, 108}
    else
      light = 0.5 + 0.5 * shade
      {250 * light, 160 * light, 64 * light}
    end
  end

  defp background(nx, ny) do
    # soft warm cream gradient, slightly darker toward the bottom
    t = ny / 100.0
    r = 246.0 - 14.0 * t
    g = 226.0 - 12.0 * t
    b = 196.0 - 10.0 * t
    _ = nx
    {r, g, b}
  end
end

defmodule SvdPaint do
  # Decompose each RGB channel once with Nx.Lapack.svd.
  def decompose(rgb) do
    {h, w, 3} = Nx.shape(rgb)
    rgb_f = Nx.as_type(rgb, :f64)

    Enum.map(0..2, fn c ->
      ch = Nx.slice(rgb_f, [0, 0, c], [h, w, 1]) |> Nx.reshape({h, w})
      Nx.Lapack.svd(ch, vectors: :full)
    end)
  end

  # Reconstruct all three channels at rank k from a decomposition.
  def paint(chans, k, h, w) do
    channels =
      Enum.map(chans, fn {u, s, vt} ->
        reconstruct(u, s, vt, k)
      end)

    Nx.stack(channels, axis: 2)
    |> Nx.reshape({h, w, 3})
    |> Nx.clip(0.0, 255.0)
    |> Nx.as_type(:u8)
  end

  defp reconstruct(u, s, vt, k) do
    n = Nx.size(s)
    k = min(k, n)
    uk = Nx.slice(u, [0, 0], [n, k])
    sk = Nx.slice(s, [0], [k])
    vk = Nx.slice(vt, [0, 0], [k, n])
    sm = Nx.multiply(Nx.eye(k, type: :f64), Nx.reshape(sk, {k, 1}))
    Nx.dot(Nx.dot(uk, sm), vk)
  end
end

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

h = 96
w = 96
out_dir = Path.join(System.tmp_dir!(), "svd_cat")
File.mkdir_p!(out_dir)

IO.puts("Rendering orange cat (#{w}x#{h})...")
cat = OrangeCat.render(h, w)
NxControl.Vision.write_bmp(Path.join(out_dir, "orange_cat_original.bmp"), cat)
IO.puts("  saved orange_cat_original.bmp")

IO.puts("Decomposing each RGB channel with Nx.Lapack.svd...")
{dec_t, chans} = :timer.tc(fn -> SvdPaint.decompose(cat) end)
IO.puts("  decomposition: #{Float.round(dec_t / 1_000_000, 1)} s (3 channels)")

for k <- [8, 16, 32, 96] do
  IO.puts("  painting at rank k=#{k}...")
  {t, painted} = :timer.tc(fn -> SvdPaint.paint(chans, k, h, w) end)
  path = Path.join(out_dir, "orange_cat_k#{k}.bmp")
  size = NxControl.Vision.write_bmp(path, painted)
  IO.puts("    #{Path.basename(path)}  (#{Float.round(t / 1000, 1)} ms, #{div(size, 1024)} KB)")
end

IO.puts("\nDone! Images saved to #{out_dir}")
