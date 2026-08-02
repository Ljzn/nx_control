# SVD-Painted Orange Cat
#
# Draw a richly detailed orange tabby cat procedurally in Nx (no external
# images), decompose each RGB channel with Nx.Lapack.svd, reconstruct at low
# ranks (the "painting"), and save the results as 24-bit BMP bitmaps.
#
# Run:  cd nx_control && mix run examples/svd_cat.exs
#
# The renderer uses NxControl.Vision-visible features: fur grain, tabby
# stripes, detailed half-closed eyes (amber iris, slit pupil, highlight),
# pink nose, whiskers, a chest patch, paws, a tail and a soft ground shadow —
# a warm, serene mood expressed through light and geometry.

defmodule OrangeCat do
  # Procedurally render a richly detailed orange tabby cat into an
  # H×W×3 RGB tensor (0..255). All geometry lives in a 100×100 space and is
  # scaled to the target resolution.

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

    # key light (top-left) and warm rim light (top-right)
    shade = light(nx, ny, 24, 26, 92.0)
    rim = light(nx, ny, 78, 30, 95.0)
    # deterministic fur grain
    grain = 0.94 + 0.06 * :math.sin(nx * 0.9 + ny * 1.7) * :math.sin(nx * 2.3 - ny * 1.3)

    in_shadow = in_ellipse(nx, ny, 50, 97.5, 44, 6.5)
    in_body = in_ellipse(nx, ny, 50, 93, 37, 27)
    in_chest = in_ellipse(nx, ny, 50, 84, 15, 13)
    in_paw_l = in_ellipse(nx, ny, 41, 106, 6.5, 4)
    in_paw_r = in_ellipse(nx, ny, 59, 106, 6.5, 4)
    in_tail = tail?(nx, ny)
    in_head = in_ellipse(nx, ny, 50, 57, 27, 25)
    in_left_ear = in_triangle(nx, ny, [[36, 10], [16, 48], [48, 44]])
    in_right_ear = in_triangle(nx, ny, [[64, 10], [84, 48], [52, 44]])
    in_inner_l = in_triangle(nx, ny, [[34, 25], [23, 46], [43, 44]])
    in_inner_r = in_triangle(nx, ny, [[66, 25], [77, 46], [57, 44]])

    cond do
      in_shadow and not (in_body or in_paw_l or in_paw_r or in_tail) ->
        darken(background(nx, ny), 0.6)

      in_left_ear or in_right_ear ->
        if (in_left_ear and in_inner_l) or (in_right_ear and in_inner_r) do
          ear_inner(ny, shade)
        else
          ear_outer(nx, ny, shade, rim, grain)
        end

      in_tail ->
        tail_color(nx, ny, shade, grain)

      in_body ->
        body_color(nx, ny, in_chest, in_paw_l, in_paw_r, shade, rim, grain)

      in_head ->
        head_color(nx, ny, shade, rim, grain)

      true ->
        background(nx, ny)
    end
  end

  defp light(x, y, lx, ly, spread) do
    max(0.0, min(1.0, 1.0 - :math.sqrt((x - lx) ** 2 + (y - ly) ** 2) / spread))
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

  # A thick diagonal tail curled to the cat's right.
  defp tail?(nx, ny) do
    rot = 0.35
    cx = 76.0
    cy = 86.0
    dx = (nx - cx) * :math.cos(rot) + (ny - cy) * :math.sin(rot)
    dy = -(nx - cx) * :math.sin(rot) + (ny - cy) * :math.cos(rot)
    (dx / 7.0) ** 2 + (dy / 16.0) ** 2 <= 1.0
  end

  # Orange fur base: stripe factor * key light * rim light * grain.
  defp fur(r, g, b, stripe, shade, rim, grain) do
    light = 0.42 + 0.58 * shade + 0.12 * rim
    {r * stripe * light * grain, g * stripe * light * grain, b * stripe * light * grain}
  end

  defp ear_outer(nx, ny, shade, rim, grain) do
    stripe = if rem(floor((nx + ny) / 5), 2) == 0, do: 0.9, else: 1.0
    fur(250, 160, 64, stripe, shade, rim, grain)
  end

  defp ear_inner(ny, shade) do
    t = max(0.0, (46.0 - ny) / 20.0)
    light = 0.5 + 0.5 * shade
    {235 * light - 30 * t, 128 * light - 20 * t, 112 * light - 16 * t}
  end

  defp tail_color(nx, ny, shade, grain) do
    stripe = if rem(floor(ny / 4), 2) == 0, do: 0.72, else: 1.0
    fur(250, 158, 62, stripe, shade, 0.05, grain)
  end

  defp body_color(nx, ny, in_chest, in_paw_l, in_paw_r, shade, rim, grain) do
    cond do
      in_chest ->
        # lighter cream chest patch with soft shading
        {250 * (0.7 + 0.3 * shade), 208 * (0.7 + 0.3 * shade), 168 * (0.7 + 0.3 * shade)}

      in_paw_l or in_paw_r ->
        light = 0.5 + 0.5 * shade
        {244 * light, 190 * light, 150 * light}

      true ->
        # vertical tabby stripes on the body
        stripe =
          cond do
            rem(floor((nx * 0.7 + ny) / 5), 2) == 0 -> 0.7
            rem(floor((nx * 0.7 + ny) / 5), 4) == 1 -> 0.85
            true -> 1.0
          end

        fur(252, 158, 62, stripe, shade, rim, grain)
    end
  end

  defp head_color(nx, ny, shade, rim, grain) do
    # forehead "M" marking
    m_stripe =
      (abs(nx - 44.5) < 1.1 or abs(nx - 50) < 1.1 or abs(nx - 55.5) < 1.1) and ny > 45.5 and
        ny < 53.5

    # cheek stripes
    cheek =
      (nx < 44 and nx > 30 and rem(floor((nx + ny) / 4), 2) == 0) or
        (nx > 56 and nx < 70 and rem(floor((nx + ny) / 4), 2) == 0)

    stripe =
      cond do
        m_stripe -> 0.62
        cheek -> 0.78
        true -> 1.0
      end

    {r, g, b} = fur(253, 160, 62, stripe, shade, rim, grain)
    add_face(nx, ny, r, g, b, shade)
  end

  defp add_face(nx, ny, r, g, b, shade) do
    eye_l = in_almond(nx, ny, 40, 52.5)
    eye_r = in_almond(nx, ny, 60, 52.5)
    eye = eye_l or eye_r
    ex = if eye_l, do: 40, else: 60

    nose = in_triangle(nx, ny, [[50, 58.5], [46.5, 61.5], [53.5, 61.5]])
    nose_hi = in_ellipse(nx, ny, 49, 59.6, 1.0, 0.7)
    muzzle_l = in_ellipse(nx, ny, 46, 62.5, 3.0, 2.0)
    muzzle_r = in_ellipse(nx, ny, 54, 62.5, 3.0, 2.0)
    mouth_l = in_ellipse(nx, ny, 47.2, 63.9, 1.8, 0.9) and ny >= 63.9
    mouth_r = in_ellipse(nx, ny, 52.8, 63.9, 1.8, 0.9) and ny >= 63.9

    cond do
      eye ->
        eye_detail(nx, ny, ex, 52.5)

      nose_hi ->
        {250, 168, 148}

      nose ->
        {228, 122, 106}

      muzzle_l or muzzle_r ->
        {242, 208, 172}

      mouth_l or mouth_r ->
        {112, 54, 40}

      whisker?(nx, ny) ->
        {232, 196, 158}

      whisker_dot(nx, ny) ->
        {170, 118, 88}

      true ->
        {r, g, b}
    end
  end

  # Half-closed almond eye with amber iris, vertical slit pupil and highlight.
  defp in_almond(nx, ny, cx, cy) do
    dx = nx - cx
    dy = ny - cy
    (dx / 4.6) ** 2 + (dy / 3.1) ** 2 <= 1.0 and ny >= cy - 2.4
  end

  defp eye_detail(nx, ny, cx, cy) do
    cond do
      # tiny specular highlight (top-left of the eye)
      (nx - (cx - 1.3)) ** 2 + (ny - (cy - 1.1)) ** 2 < 0.85 ** 2 -> {255, 250, 235}
      # dark vertical pupil
      ((nx - cx) / 0.85) ** 2 + ((ny - cy) / 1.7) ** 2 <= 1.0 -> {20, 14, 10}
      # amber iris
      true -> {255, 168, 66}
    end
  end

  defp whisker?(nx, ny) do
    segs = [
      {41, 60.5, 24, 57.5},
      {41, 61.8, 23, 61.5},
      {41, 63, 25, 65.5},
      {41, 64, 27, 68.5},
      {59, 60.5, 76, 57.5},
      {59, 61.8, 77, 61.5},
      {59, 63, 75, 65.5},
      {59, 64, 73, 68.5}
    ]

    Enum.any?(segs, fn {x1, y1, x2, y2} -> near_segment(nx, ny, x1, y1, x2, y2, 0.65) end)
  end

  defp whisker_dot(nx, ny) do
    Enum.any?([{43.5, 60.5}, {42.5, 62}, {56.5, 60.5}, {57.5, 62}], fn {dx, dy} ->
      (nx - dx) ** 2 + (ny - dy) ** 2 < 0.8 ** 2
    end)
  end

  defp near_segment(px, py, x1, y1, x2, y2, tol) do
    vx = x2 - x1
    vy = y2 - y1
    len2 = vx * vx + vy * vy
    t = if len2 == 0, do: 0.0, else: max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / len2))
    qx = x1 + t * vx
    qy = y1 + t * vy
    (px - qx) ** 2 + (py - qy) ** 2 <= tol * tol
  end

  defp background(nx, ny) do
    # soft warm cream gradient with a subtle vignette
    t = ny / 100.0
    edge = max(abs(nx - 50) / 50.0, t)
    vignette = 1.0 - 0.08 * edge * edge
    {r, g, b} = {246.0 - 14.0 * t, 226.0 - 12.0 * t, 196.0 - 10.0 * t}
    {r * vignette, g * vignette, b * vignette}
  end

  defp darken({r, g, b}, f), do: {r * f, g * f, b * f}
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

h = 128
w = 128
out_dir = Path.join(System.tmp_dir!(), "svd_cat")
File.mkdir_p!(out_dir)

IO.puts("Rendering orange cat (#{w}x#{h})...")
cat = OrangeCat.render(h, w)
NxControl.Vision.write_bmp(Path.join(out_dir, "orange_cat_original.bmp"), cat)
IO.puts("  saved orange_cat_original.bmp")

IO.puts("Decomposing each RGB channel with Nx.Lapack.svd...")
{dec_t, chans} = :timer.tc(fn -> SvdPaint.decompose(cat) end)
IO.puts("  decomposition: #{Float.round(dec_t / 1_000_000, 1)} s (3 channels)")

for k <- [8, 16, 32, 64, 128] do
  IO.puts("  painting at rank k=#{k}...")
  {t, painted} = :timer.tc(fn -> SvdPaint.paint(chans, k, h, w) end)
  path = Path.join(out_dir, "orange_cat_k#{k}.bmp")
  size = NxControl.Vision.write_bmp(path, painted)
  IO.puts("    #{Path.basename(path)}  (#{Float.round(t / 1000, 1)} ms, #{div(size, 1024)} KB)")
end

IO.puts("\nDone! Images saved to #{out_dir}")
