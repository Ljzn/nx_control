defmodule NxControl.SteadyState do
  @moduledoc """
  Steady-state error analysis for unity feedback systems.

  For a unity feedback system with open-loop transfer function G(s),
  the steady-state error depends on the system type (number of
  integrators in G(s)) and the input signal.

  ## Error constants

  | Input       | Type 0      | Type 1      | Type 2      |
  |-------------|-------------|-------------|-------------|
  | Step `1/s`  | `1/(1+Kp)`  | 0           | 0           |
  | Ramp `1/s²` | ∞           | `1/Kv`      | 0           |
  | Parabola `1/s³` | ∞      | ∞           | `1/Ka`      |

  where:

      Kp = lim_{s→0} G(s)
      Kv = lim_{s→0} s·G(s)
      Ka = lim_{s→0} s²·G(s)
  """

  alias NxControl.TransferFunction

  @doc """
  Returns the system type (number of integrators at s = 0).

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1], [1, 3, 2])
      iex> NxControl.SteadyState.system_type(tf)
      0

      iex> tf = NxControl.TransferFunction.new([1], [1, 1, 0])
      iex> NxControl.SteadyState.system_type(tf)
      1
  """
  def system_type(tf) do
    tf.den
    |> Nx.to_flat_list()
    |> Enum.reverse()
    |> Enum.take_while(&(abs(&1) < 1.0e-15))
    |> length()
  end

  @doc """
  Position error constant Kp = lim_{s→0} G(s).

  ## Examples

      iex> tf = NxControl.TransferFunction.new([5], [1, 3, 2])
      iex> NxControl.SteadyState.kp(tf)
      2.5
  """
  def kp(tf) do
    s0 = Nx.tensor(0, type: {:c, 128})
    Nx.to_number(Nx.real(TransferFunction.evalf(tf, s0)))
  end

  @doc """
  Velocity error constant Kv = lim_{s→0} s·G(s).

  For a type 1 system, this is equivalent to evaluating
  num(0) / (den/s)(0), where the common s factor has been
  cancelled.

  ## Examples

      iex> tf = NxControl.TransferFunction.new([5], [1, 1, 0])
      iex> NxControl.SteadyState.kv(tf)
      5.0
  """
  def kv(tf) do
    kv_from_limit(tf)
  end

  @doc """
  Acceleration error constant Ka = lim_{s→0} s²·G(s).

  ## Examples

      iex> tf = NxControl.TransferFunction.new([5], [1, 1, 0, 0])
      iex> NxControl.SteadyState.ka(tf)
      5.0
  """
  def ka(tf) do
    ka_from_limit(tf)
  end

  @doc """
  Steady-state error for a unit step input r(t) = 1.

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1], [1, 1])
      iex> NxControl.SteadyState.step_error(tf)
      0.5

      iex> tf = NxControl.TransferFunction.new([1], [1, 1, 0])
      iex> NxControl.SteadyState.step_error(tf)
      0.0
  """
  def step_error(tf) do
    if system_type(tf) >= 1 do
      0.0
    else
      1.0 / (1.0 + kp(tf))
    end
  end

  @doc """
  Steady-state error for a unit ramp input r(t) = t.

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1], [1, 3, 2])
      iex> NxControl.SteadyState.ramp_error(tf)
      :infinity

      iex> tf = NxControl.TransferFunction.new([5], [1, 1, 0])
      iex> NxControl.SteadyState.ramp_error(tf)
      0.2
  """
  def ramp_error(tf) do
    case system_type(tf) do
      t when t >= 2 -> 0.0
      1 -> 1.0 / kv(tf)
      _ -> :infinity
    end
  end

  @doc """
  Steady-state error for a unit parabolic input r(t) = t²/2.

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1], [1, 3, 2])
      iex> NxControl.SteadyState.parabolic_error(tf)
      :infinity

      iex> tf = NxControl.TransferFunction.new([5], [1, 1, 0, 0])
      iex> NxControl.SteadyState.parabolic_error(tf)
      0.2
  """
  def parabolic_error(tf) do
    case system_type(tf) do
      t when t >= 3 -> 0.0
      2 -> 1.0 / ka(tf)
      _ -> :infinity
    end
  end

  # Kv = lim_{s→0} s·G(s) = num(0) / (den/s)(0)
  defp kv_from_limit(tf) do
    t = system_type(tf)

    if t >= 1 do
      den_list = Nx.to_flat_list(tf.den)
      num_list = Nx.to_flat_list(tf.num)
      den_rest = strip_trailing(den_list, 1)
      num0 = List.last(num_list) || 0.0
      den0 = List.last(den_rest) || 0.0
      num0 / den0
    else
      0.0
    end
  end

  # Ka = lim_{s→0} s²·G(s) = num(0) / (den/s²)(0)
  defp ka_from_limit(tf) do
    t = system_type(tf)

    if t >= 2 do
      den_list = Nx.to_flat_list(tf.den)
      num_list = Nx.to_flat_list(tf.num)
      den_rest = strip_trailing(den_list, 2)
      num0 = List.last(num_list) || 0.0
      den0 = List.last(den_rest) || 0.0
      num0 / den0
    else
      0.0
    end
  end

  defp strip_trailing(list, n) do
    list |> Enum.reverse() |> Enum.drop(n) |> Enum.reverse()
  end
end
