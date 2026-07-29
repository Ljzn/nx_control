defmodule NxControl.FrequencyRespTest do
  use ExUnit.Case, async: true

  alias NxControl.TransferFunction
  alias NxControl.FrequencyResp

  test "bode_data returns correct number of points" do
    tf = TransferFunction.new([1], [1, 2, 1])
    {freqs, mag, phase} = FrequencyResp.bode_data(tf, n_points: 50)
    assert Nx.size(freqs) == 50
    assert Nx.size(mag) == 50
    assert Nx.size(phase) == 50
  end

  test "bode_data default options" do
    tf = TransferFunction.new([1], [1, 2, 1])
    {freqs, mag, phase} = FrequencyResp.bode_data(tf)
    assert Nx.size(freqs) == 200
    assert Nx.size(mag) == 200
    assert Nx.size(phase) == 200
  end

  test "nyquist_data returns correct number of points" do
    tf = TransferFunction.new([1], [1, 2, 1])
    {real, imag} = FrequencyResp.nyquist_data(tf, n_points: 50)
    assert Nx.size(real) == 50
    assert Nx.size(imag) == 50
  end

  test "margins returns four values" do
    tf = TransferFunction.new([1], [1, 1])
    {gm, gp, pm_f, pm_p} = FrequencyResp.margins(tf)
    assert is_number(gm) or gm == :infinity
    assert is_number(gp) or gp == nil
    assert is_number(pm_f) or pm_f == nil
    assert is_number(pm_p) or pm_p == nil
  end

  test "DC gain at ω=0 for type 0 system" do
    tf = TransferFunction.new([5], [1, 3, 2])
    {freqs, mag, _phase} = FrequencyResp.bode_data(tf, n_points: 200, freq_range: {-2, 4})
    # At lowest frequency (ω → 0), |G(jω)| → Kp = 2.5 → 20*log10(2.5) ≈ 7.96 dB
    dc_gain = Nx.to_number(mag[0])
    assert_in_delta dc_gain, 20 * :math.log10(2.5), 1.0
  end
end
