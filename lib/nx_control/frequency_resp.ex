defmodule NxControl.FrequencyResp do
  @moduledoc """
  Frequency response analysis — data generation for Bode and Nyquist plots.

  Returns pure data (frequency / magnitude / phase / real / imaginary parts)
  with no dependency on any plotting library. The data can be plotted directly
  with VegaLite in Livebook or exported to external tools.

  ## Bode plot

  The Bode plot consists of two subplots:

    * Magnitude: gain (dB) vs angular frequency (log scale)
    * Phase: phase (degrees) vs angular frequency (log scale)

  ## Nyquist plot

  The Nyquist plot traces Re(G(jω)) vs Im(G(jω)) for ω from 0 to +∞,
  used for the Nyquist stability criterion.
  """

  alias NxControl.TransferFunction

  @doc """
  Generates Bode plot data.

  Returns `{freqs, mag_db, phase_deg}`, three Nx vectors.
  The phase is unwrapped so it accumulates continuously along the frequency
  axis, matching MATLAB / scipy's `bode` (e.g. a three-pole system reaches
  -270°).

  ## Options

    * `:n_points` — number of frequency points. Default `200`.
    * `:freq_range` — `{log10(w_start), log10(w_end)}`. Default `{-2, 4}` (i.e. 10⁻² ~ 10⁴ rad/s).
    * `:method` — pole method, forwarded to `TransferFunction.poles/2`. Default `:companion`.

  ## Examples

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
  Generates Nyquist plot data.

  Returns `{real, imag}`, two Nx vectors tracing G(jω) in the complex plane.

  ## Options

    Same as `bode_data/2`.

  ## Examples

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
  Computes gain margin and phase margin.

  Returns `{gm_mag, gm_phase, pm_freq, pm_phase}`:

    * `gm_mag` — gain margin (linear), the reciprocal of \|G(jω)\| at the phase crossover
    * `gm_phase` — phase (degrees) at the phase crossover
    * `pm_freq` — angular frequency (rad/s) at the gain crossover
    * `pm_phase` — phase margin (degrees), the distance of the phase from -180°

  If a crossover is not found, the corresponding value is `:infinity` or `nil`.

  ## Options

    Same as `bode_data/2`, plus:

    * `:gm_tol` — tolerance for locating the phase crossover. Default `0.5` degrees.
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

    # gain margin: magnitude at the phase crossover (-180°)
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

    # phase margin: phase offset at the 0 dB (magnitude = 1) gain crossover
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
