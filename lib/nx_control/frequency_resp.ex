defmodule NxControl.FrequencyResp do
  @moduledoc """
  频率响应分析 — Bode 图、Nyquist 图的数据生成。

  返回纯数据（频率/幅值/相角/实部/虚部），不依赖任何绘图库。
  数据可直接用于 Livebook 的 VegaLite 绘图或导出到外部工具。

  ## Bode 图

  Bode 图由两张子图组成：

    * 幅频特性：增益（dB）vs 角频率（对数坐标）
    * 相频特性：相位（度）vs 角频率（对数坐标）

  ## Nyquist 图

  Nyquist 图绘制 Re(G(jω)) vs Im(G(jω))，ω 从 0 到 +∞。
  用于 Nyquist 稳定判据。
  """

  alias NxControl.TransferFunction

  @doc """
  生成 Bode 图数据。

  返回 `{freqs, mag_db, phase_deg}`，三个 Nx 向量。
  相位已解包（phase unwrapping），会在频率轴方向上连续累积，
  与 MATLAB / scipy 的 `bode` 结果一致（例如三极点系统达到 -270°）。

  ## Options

    * `:n_points` — 频率点数量。默认 `200`。
    * `:freq_range` — `{log10(w_start), log10(w_end)}`。默认 `{-2, 4}`（即 10⁻² ~ 10⁴ rad/s）。
    * `:method` — 极点求法，传给 `TransferFunction.poles/2`。默认 `:companion`。

  ## 示例

      tf = NxControl.TransferFunction.new([1], [1, 2, 1])
      {freqs, mag_db, phase_deg} = NxControl.FrequencyResp.bode_data(tf)
  """
  def bode_data(tf, opts \\ []) do
    freqs = gen_freq_points(opts)

    {mag, phase} =
      Enum.reduce(freqs, {[], []}, fn w, {mag_acc, phase_acc} ->
        s = Nx.tensor(Complex.new(0, w), type: {:c, 128})
        g = TransferFunction.evalf(tf, s)

        re = Nx.to_number(Nx.real(g))
        im = Nx.to_number(Nx.imag(g))

        mag_val = 20 * :math.log10(:math.sqrt(re * re + im * im) + 1.0e-30)
        phase_val = :math.atan2(im, re) * 180 / :math.pi()

        {[mag_val | mag_acc], [phase_val | phase_acc]}
      end)

    {
      Nx.tensor(freqs, type: :f64),
      Nx.tensor(Enum.reverse(mag), type: :f64),
      Nx.tensor(phase |> Enum.reverse() |> unwrap_phase(), type: :f64)
    }
  end

  # Unwraps the raw atan2 phase (bounded to [-180, 180]) so it accumulates
  # continuously across frequency, matching MATLAB and scipy. Each pole or zero
  # contributes -90° / +90° at high frequency, so e.g. a 3-pole system reaches
  # -270° instead of being wrapped to +90°.
  defp unwrap_phase([first | rest]) do
    {unwrapped, _} =
      Enum.map_reduce(rest, first, fn phase, prev ->
        adjustment = round((phase - prev + 180.0) / 360.0) * 360.0
        unwrapped = phase - adjustment
        {unwrapped, unwrapped}
      end)

    [first | unwrapped]
  end

  @doc """
  生成 Nyquist 图数据。

  返回 `{real, imag}`，两个 Nx 向量，构成 G(jω) 在复平面上的轨迹。

  ## Options

    同 `bode_data/2`。

  ## 示例

      tf = NxControl.TransferFunction.new([1], [1, 1])
      {real, imag} = NxControl.FrequencyResp.nyquist_data(tf)
  """
  def nyquist_data(tf, opts \\ []) do
    freqs = gen_freq_points(opts)

    {reals, imags} =
      Enum.reduce(freqs, {[], []}, fn w, {re_acc, im_acc} ->
        s = Nx.tensor(Complex.new(0, w), type: {:c, 128})
        g = TransferFunction.evalf(tf, s)

        re = Nx.to_number(Nx.real(g))
        im = Nx.to_number(Nx.imag(g))

        {[re | re_acc], [im | im_acc]}
      end)

    {
      Nx.tensor(Enum.reverse(reals), type: :f64),
      Nx.tensor(Enum.reverse(imags), type: :f64)
    }
  end

  @doc """
  计算幅值裕度（Gain Margin）和相角裕度（Phase Margin）。

  返回 `{gm_mag, gm_phase, pm_freq, pm_phase}`：

    * `gm_mag` — 幅值裕度（线性），即相角交叉点处 \|G(jω)\| 的倒数
    * `gm_phase` — 相角交叉点处的相角（度）
    * `pm_freq` — 增益交叉点处的角频率（rad/s）
    * `pm_phase` — 增益交叉点处的相角距离 -180° 的差值（度）

  若交叉点不存在，对应值为 `:infinity` 或 `nil`。

  ## Options

    同 `bode_data/2`，额外：

    * `:gm_tol` — 搜索相角交叉点的容差，默认 `0.5` 度。
  """
  def margins(tf, opts \\ []) do
    n = opts[:n_points] || 2000
    freq_range = opts[:freq_range] || {-2, 6}
    gm_tol = opts[:gm_tol] || 0.5

    freqs = gen_freq_points(n: n, freq_range: freq_range)

    {mag_list, phase_list} =
      Enum.reduce(freqs, {[], []}, fn w, {mag_acc, phase_acc} ->
        s = Nx.tensor(Complex.new(0, w), type: {:c, 128})
        g = TransferFunction.evalf(tf, s)
        re = Nx.to_number(Nx.real(g))
        im = Nx.to_number(Nx.imag(g))
        mag = :math.sqrt(re * re + im * im)
        phase = :math.atan2(im, re) * 180 / :math.pi()
        {[mag | mag_acc], [phase | phase_acc]}
      end)

    mag_list = Enum.reverse(mag_list)
    phase_list = Enum.reverse(phase_list)

    # 增益裕度：相角 = -180° 处的增益
    gm_entry =
      Enum.zip(freqs, phase_list)
      |> Enum.zip(mag_list)
      |> Enum.find(fn {{_w, phase}, _mag} -> abs(phase + 180) <= gm_tol end)

    {gm_mag, gm_phase} =
      if gm_entry do
        {{w_gm, phase_gm}, mag_gm} = gm_entry
        {1.0 / mag_gm, phase_gm}
      else
        {:infinity, nil}
      end

    # 相角裕度：增益 = 0 dB (magnitude = 1) 处的相位差
    pm_entry =
      Enum.zip(freqs, mag_list)
      |> Enum.zip(phase_list)
      |> Enum.find(fn {{_w, mag}, _phase} -> mag <= 1.0 end)

    {pm_freq, pm_phase} =
      if pm_entry do
        {{w_pm, _mag}, phase_pm} = pm_entry
        {w_pm, phase_pm + 180}
      else
        {nil, nil}
      end

    {gm_mag, gm_phase, pm_freq, pm_phase}
  end

  defp gen_freq_points(opts) do
    n = opts[:n_points] || 200
    {w0, w1} = opts[:freq_range] || {-2, 4}

    if n == 1 do
      [10.0 ** w0]
    else
      for i <- 0..(n - 1), do: 10 ** (w0 + (w1 - w0) * i / (n - 1))
    end
  end
end
