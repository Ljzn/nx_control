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
end
