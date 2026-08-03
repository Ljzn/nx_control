defmodule NxControl.VisionTest do
  use ExUnit.Case, async: true

  alias NxControl.Vision

  defp solid(r, g, b, h \\ 8, w \\ 8) do
    Nx.broadcast(Nx.tensor([r, g, b], type: :f64), {h, w, 3})
  end

  test "write_bmp/read_bmp round-trips exactly" do
    img = Nx.iota({5, 7, 3}, type: :f64) |> Nx.as_type(:u8)
    path = Path.join(System.tmp_dir!(), "vision_rt_test.bmp")
    Vision.write_bmp(path, img)

    img2 = Vision.read_bmp(path)
    assert Nx.shape(img2) == {5, 7, 3}
    assert Nx.to_flat_list(img2) == Nx.to_flat_list(img)
  end

  test "read_bmp rejects malformed files" do
    path = Path.join(System.tmp_dir!(), "vision_bad.bmp")
    File.write!(path, "not a bmp file at all")
    assert_raise MatchError, fn -> Vision.read_bmp(path) end
  end

  test "analyze reports per-channel statistics" do
    img = solid(250, 160, 60)
    report = Vision.analyze(img, max_size: 8, cells: 4)
    assert report.colors.r.mean == 250.0
    assert report.colors.g.mean == 160.0
    assert report.colors.b.mean == 60.0
    assert report.colors.r.sd == 0.0
  end

  test "analyze detects orange hue" do
    img = solid(250, 160, 60)
    report = Vision.analyze(img, max_size: 8, cells: 4)
    names = Enum.map(report.hues, &elem(&1, 0))
    assert "orange" in names
    assert elem(hd(report.hues), 1) > 0.9
  end

  test "analyze detects flat vs textured regions via texture flatness" do
    flat = solid(120, 120, 120, 16, 16)
    r_flat = Vision.analyze(flat, max_size: 16, cells: 4)
    assert r_flat.texture.mean_flatness > 0.99

    # gradient image has structure
    grad =
      Nx.iota({16, 16}, type: :f64)
      |> Nx.divide(Nx.tensor(16.0, type: :f64))
      |> Nx.reshape({16, 16, 1})
      |> Nx.broadcast({16, 16, 3})

    r_grad = Vision.analyze(grad, max_size: 16, cells: 4)
    assert r_grad.texture.mean_flatness < r_flat.texture.mean_flatness
  end

  test "edge energy is ~zero on a flat image and positive on a gradient" do
    flat = solid(80, 80, 80, 16, 16)
    r_flat = Vision.analyze(flat, max_size: 16, cells: 4)
    assert r_flat.edges.total < 1.0e-6

    grad =
      Nx.iota({16, 16}, type: :f64)
      |> Nx.divide(Nx.tensor(16.0, type: :f64))
      |> Nx.reshape({16, 16, 1})
      |> Nx.broadcast({16, 16, 3})

    r_grad = Vision.analyze(grad, max_size: 16, cells: 4)
    assert r_grad.edges.total > 0.0
  end

  test "to_text renders the report" do
    img = solid(250, 160, 60)
    report = Vision.analyze(img, max_size: 8, cells: 4)
    text = Vision.to_text(report)

    assert text =~ "IMAGE"
    assert text =~ "COLOR"
    assert text =~ "HUES"
    assert text =~ "GRID"
    assert text =~ "SVD"
  end

  test "ascii returns a color-letter grid" do
    img = solid(250, 160, 60, 16, 16)
    text = Vision.ascii(img, cells: 4, max_size: 16)
    lines = String.split(text, "\n") |> Enum.filter(&(&1 != ""))
    assert length(lines) > 0
    assert Enum.all?(lines, fn line -> String.length(line) == 4 end)
  end

  test "compress rank-k approximates the original" do
    img = solid(200, 100, 50, 8, 8)
    full = Vision.compress(img, k: 8)
    # full rank should reproduce the image
    err_full = Nx.to_number(Nx.abs(Nx.subtract(full, Nx.as_type(img, :f64))) |> Nx.sum())
    assert err_full < 1.0e-2
  end

  test "mosaic preserves shape" do
    img = solid(200, 100, 50, 20, 24)
    mos = Vision.mosaic(img, cells: 4)
    assert Nx.shape(mos) == {20, 24, 3}
  end

  test "luminance matches the standard weights" do
    img = Nx.tensor([[100.0, 200.0, 300.0]], type: :f64) |> Nx.reshape({1, 1, 3})
    lum = Vision.luminance(img) |> Nx.reshape({}) |> Nx.to_number()
    assert_in_delta lum, 0.299 * 100 + 0.587 * 200 + 0.114 * 300, 1.0e-4
  end

  # ─────────────────────────────────────────────
  # Region / zoom
  # ─────────────────────────────────────────────

  test "analyze with region reports region shape and bounds" do
    img = solid(100, 100, 100, 16, 24)
    report = Vision.analyze(img, region: {2, 10, 4, 20}, cells: 4, max_size: 16)
    assert report.shape == {8, 16}
    assert report.region == {2, 10, 4, 20}
  end

  test "region is nil for whole-image analysis" do
    report = Vision.analyze(solid(100, 100, 100), cells: 4, max_size: 8)
    assert report.region == nil
    assert report.shape == {8, 8}
  end

  test "region bounds are clamped and swapped" do
    img = solid(100, 100, 100, 8, 8)
    # inverted y / out-of-range x
    report = Vision.analyze(img, region: {6, 2, -5, 100}, cells: 4, max_size: 8)
    assert report.region == {2, 6, 0, 8}
    assert report.shape == {4, 8}
  end

  test "analyzing a region keeps the report pipeline working" do
    img = solid(250, 160, 60, 16, 16)
    report = Vision.analyze(img, region: {4, 12, 4, 12}, cells: 4, max_size: 8)
    names = Enum.map(report.hues, &elem(&1, 0))
    assert "orange" in names
    assert report.edges.total < 1.0e-6
  end

  test "quadrants returns grid x grid sub-reports with regions" do
    img = solid(120, 120, 120, 8, 8)
    reports = Vision.quadrants(img, grid: 2, cells: 2, max_size: 8)
    assert length(reports) == 4

    assert Enum.map(reports, & &1.region) == [
             {0, 4, 0, 4},
             {0, 4, 4, 8},
             {4, 8, 0, 4},
             {4, 8, 4, 8}
           ]

    assert Enum.all?(reports, &(&1.shape == {4, 4}))
  end

  test "ascii respects region and cells" do
    img = solid(200, 100, 50, 16, 16)
    text = Vision.ascii(img, region: {4, 12, 4, 12}, cells: 4, max_size: 16)
    lines = String.split(text, "\n") |> Enum.filter(&(&1 != ""))
    assert Enum.all?(lines, fn line -> String.length(line) == 4 end)
  end

  test "to_text renders the region line when present" do
    img = solid(200, 100, 50, 16, 16)
    text = Vision.to_text(Vision.analyze(img, region: {2, 8, 2, 8}, cells: 4, max_size: 8))
    assert text =~ "REGION y=2-8 x=2-8"
  end

  # ─────────────────────────────────────────────
  # Tone classification
  # ─────────────────────────────────────────────

  test "saturated red with g < b does not crash and maps to a wheel bin" do
    # regression: hue -40 -> trunc(-40/30) = -1 used to produce a negative
    # bin via rem/2, making Enum.find return nil and crash
    img = solid(200, 50, 150)
    report = Vision.analyze(img, max_size: 8, cells: 4)
    names = Enum.map(report.hues, &elem(&1, 0))
    assert "red-violet" in names
    assert elem(hd(report.hues), 1) > 0.9
  end

  test "sepia tones are reported as warm, not red" do
    img = solid(170, 150, 140)
    report = Vision.analyze(img, max_size: 8, cells: 4)
    names = Enum.map(report.hues, &elem(&1, 0))
    assert "warm" in names
    refute "red" in names
    refute "orange" in names
  end

  test "pure gray is reported as neutral" do
    img = solid(128, 128, 128)
    report = Vision.analyze(img, max_size: 8, cells: 4)
    names = Enum.map(report.hues, &elem(&1, 0))
    assert "neutral" in names
  end

  test "hue histogram reports configurable top-N" do
    quad =
      fn rgb ->
        Nx.broadcast(Nx.tensor(rgb, type: :f64), {4, 4, 3})
      end

    img =
      Nx.concatenate(
        [
          Nx.concatenate([quad.([250, 160, 60]), quad.([200, 50, 150])], axis: 1),
          Nx.concatenate([quad.([170, 150, 140]), quad.([128, 128, 128])], axis: 1)
        ],
        axis: 0
      )

    report = Vision.analyze(img, max_size: 8, cells: 4, top: 4)
    assert length(report.hues) == 4
  end

  # ─────────────────────────────────────────────
  # Sample / mask / luma
  # ─────────────────────────────────────────────

  test "sample returns the mean of a local patch" do
    img = solid(10, 20, 30)
    assert Vision.sample(img, 4, 4, radius: 1) == {10.0, 20.0, 30.0}
  end

  test "sample clamps at the borders" do
    img = solid(10, 20, 30)
    assert Vision.sample(img, 0, 0, radius: 2) == {10.0, 20.0, 30.0}
  end

  test "mask returns a binary tensor at the threshold" do
    img = solid(200, 200, 200, 8, 8)
    m = Vision.mask(img, threshold: 128)
    assert Nx.shape(m) == {8, 8}
    assert Nx.to_number(Nx.sum(m)) == 64

    dark = Vision.mask(solid(50, 50, 50, 8, 8), threshold: 128)
    assert Nx.to_number(Nx.sum(dark)) == 0
  end

  test "mask_ascii renders # and . rows" do
    text = Vision.mask_ascii(solid(200, 200, 200, 4, 4), threshold: 128, rows: 2, cols: 2)
    lines = String.split(text, "\n") |> Enum.filter(&(&1 != ""))
    assert Enum.all?(lines, fn line -> line == "##" end)
  end

  test "ascii mode luma returns graded characters" do
    img = solid(255, 255, 255, 16, 16)
    text = Vision.ascii(img, mode: :luma, cells: 4, max_size: 16)
    lines = String.split(text, "\n") |> Enum.filter(&(&1 != ""))
    assert Enum.all?(lines, fn line -> String.length(line) == 4 end)
    assert String.contains?(text, "&")
  end

  test "ascii mode hue marks warm gray with w" do
    img = solid(180, 165, 150, 8, 8)
    text = Vision.ascii(img, cells: 2, max_size: 8)
    assert String.contains?(text, "w")
    assert not String.contains?(text, "g")
  end

  test "detail option controls resolution" do
    img = solid(100, 100, 100, 80, 80)
    coarse = Vision.analyze(img, detail: :coarse, cells: 2)
    native = Vision.analyze(img, detail: :native, cells: 2)
    assert coarse.svd.size == {40, 40}
    assert native.svd.size == {80, 80}
  end
end
