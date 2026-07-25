defmodule NxControl.TransferFunction do
  @moduledoc """
  Transfer function representation `G(s) = num(s) / den(s)`.

  A transfer function models a linear time-invariant (LTI) system as the
  ratio of two polynomials in the Laplace variable `s`:

      G(s) = b₀·sᵐ + b₁·sᵐ⁻¹ + ... + bₘ
             ─────────────────────────────
             a₀·sⁿ + a₁·sⁿ⁻¹ + ... + aₙ

  Numerator and denominator coefficients are stored as `Nx.Tensor` vectors
  in descending power order.

  ## Stability analysis

  NxControl provides two complementary methods:

    * **Routh-Hurwitz** (`stable?/1`) — examines the characteristic
      polynomial coefficients to determine stability without computing
      roots. Fast and numerically robust.

    * **Pole-based** (`pole_stable?/1`) — computes all poles via the
      DKA (Durand-Kerner-Aberth) polynomial root finder and checks
      that all real parts are negative.

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1], [1, 3, 2])
      iex> NxControl.TransferFunction.stable?(tf)
      true
      iex> NxControl.TransferFunction.poles(tf)
      #Nx.Tensor<c128[2] [-1.0+0.0i, -2.0+0.0i]>
  """

  defstruct [:num, :den]

  @doc """
  Creates a new transfer function.

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1], [1, 2, 1])
      iex> tf.num
      #Nx.Tensor<f64[1] [1.0]>
      iex> tf.den
      #Nx.Tensor<f64[3] [1.0, 2.0, 1.0]>
  """
  def new(num, den) when is_list(num) and is_list(den) do
    %__MODULE__{
      num: Nx.tensor(num, type: :f64),
      den: Nx.tensor(den, type: :f64)
    }
  end

  def new(num, den) when is_struct(num, Nx.Tensor) and is_struct(den, Nx.Tensor) do
    %__MODULE__{num: num, den: den}
  end

  @doc """
  Returns the poles of the transfer function (roots of denominator).

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1], [1, 3, 2])
      iex> NxControl.TransferFunction.poles(tf)
      #Nx.Tensor<c128[2] [-1.0+0.0i, -2.0+0.0i]>
  """
  def poles(tf) do
    dka_roots(tf.den)
  end

  @doc """
  Returns the zeros of the transfer function (roots of numerator).

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1, 1], [1, 3, 2])
      iex> NxControl.TransferFunction.zeros(tf)
      #Nx.Tensor<c128[1] [-1.0+0.0i]>
  """
  def zeros(tf) do
    dka_roots(tf.num)
  end

  @doc """
  Checks if the system is stable by verifying all poles have negative real parts.

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1], [1, 3, 2])
      iex> NxControl.TransferFunction.pole_stable?(tf)
      true

      iex> tf = NxControl.TransferFunction.new([1], [1, -1, -6])
      iex> NxControl.TransferFunction.pole_stable?(tf)
      false
  """
  def pole_stable?(tf) do
    p = poles(tf)
    real_parts = Nx.real(p)
    Nx.to_number(Nx.all(Nx.less(real_parts, 0))) == 1
  end

  @doc """
  Checks if the system is stable using the Routh-Hurwitz criterion.
  All coefficients of the characteristic polynomial must be positive
  and the Routh array's first column must have no sign changes.

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1], [1, 2, 1])
      iex> NxControl.TransferFunction.stable?(tf)
      true

      iex> tf = NxControl.TransferFunction.new([1], [1, -1, -6])
      iex> NxControl.TransferFunction.stable?(tf)
      false
  """
  def stable?(tf) do
    den = Nx.to_flat_list(tf.den)

    sign = if hd(den) > 0, do: 1, else: -1

    if Enum.any?(den, fn c -> c * sign <= 0 end) do
      false
    else
      routh_array = build_routh_array(den)
      first_col = Enum.map(routh_array, &List.first/1)
      Enum.all?(first_col, &(&1 * sign > 0))
    end
  end

  @doc """
  Evaluates the transfer function at a given complex frequency s.

  G(s) = num(s) / den(s)

  ## Examples

      iex> tf = NxControl.TransferFunction.new([1], [1, 2, 1])
      iex> s = Nx.tensor(0, type: {:c, 128})
      iex> NxControl.TransferFunction.evalf(tf, s)
      #Nx.Tensor<c128 1.0+0.0i>
  """
  def evalf(tf, s) do
    num_val = polyval_c128(tf.num, s)
    den_val = polyval_c128(tf.den, s)
    Nx.divide(num_val, den_val)
  end

  # --- DKA (Durand-Kerner-Aberth) polynomial root finder ---

  defp dka_roots(coefs) do
    n = Nx.size(coefs) - 1

    cond do
      n <= 0 ->
        Nx.tensor([])

      n == 1 ->
        a0 = Nx.slice(coefs, [1], [1]) |> Nx.reshape({})
        a1 = Nx.slice(coefs, [0], [1]) |> Nx.reshape({})
        Nx.negate(a0) |> Nx.divide(a1) |> Nx.reshape({1}) |> Nx.as_type({:c, 128})

      true ->
        a0 = Nx.to_number(coefs[0])
        poly = Nx.divide(coefs, a0)
        poly_list = Nx.to_flat_list(poly)

        # Aberth initial guess
        center = -Enum.at(poly_list, 1, 0.0) / n
        max_abs = Enum.reduce(poly_list, 0.0, fn c, acc -> max(abs(c), acc) end)
        radius = 1.0 + max_abs

        init_vals =
          for j <- 0..(n - 1) do
            angle = 2 * :math.pi() * j / n + 0.5 / n
            real = center + radius * :math.cos(angle)
            imag = radius * :math.sin(angle)
            Complex.new(real, imag)
          end

        z = Nx.tensor(init_vals, type: {:c, 128})
        dka_iterate(z, poly, n, 100)
    end
  end

  defp dka_iterate(z, poly, n, max_iter) do
    tol = 1.0e-12

    {z, _} =
      Enum.reduce_while(1..max_iter, {z, false}, fn _iter, {z_acc, _done} ->
        # Evaluate polynomial at all roots: P(z_j) for all j
        p_vals = polyval_c128(poly, z_acc)

        # Build difference matrix: diff_{j,k} = z_j - z_k
        z_col = Nx.reshape(z_acc, {n, 1})
        z_row = Nx.reshape(z_acc, {1, n})
        diff = Nx.subtract(z_col, z_row)

        # Set diagonal to 1 to avoid zero denominator
        eye = Nx.eye(n, type: {:c, 128})
        diff_safe = Nx.select(Nx.equal(eye, 1), Nx.tensor(1.0, type: {:c, 128}), diff)

        # denom_j = ∏_{k≠j} (z_j - z_k)
        denom = Nx.product(diff_safe, axes: [1])

        # Δz_j = -P(z_j) / denom_j
        delta = Nx.negate(p_vals) |> Nx.divide(denom)
        z_new = Nx.add(z_acc, delta)

        max_delta = Nx.to_number(Nx.reduce_max(Nx.abs(delta)))

        if max_delta < tol do
          {:halt, {z_new, true}}
        else
          {:cont, {z_new, max_delta}}
        end
      end)

    z
  end

  # Evaluates a polynomial at a complex value s, or element-wise at each
  # element of tensor s (same as Horner's method on each element).
  defp polyval_c128(coefs, s) do
    # coefs: [a_n, a_{n-1}, ..., a_0], type f64
    # s: scalar or tensor of type c128
    # Horner: (((a_n * s + a_{n-1}) * s + a_{n-2}) * s + ... + a_0)
    n = Nx.size(coefs) - 1
    result = Nx.as_type(coefs[0], {:c, 128})

    result =
      for i <- 1..n//1, reduce: result do
        acc -> Nx.add(Nx.multiply(acc, s), Nx.as_type(coefs[i], {:c, 128}))
      end

    result
  end

  # --- Routh-Hurwitz utility ---

  defp build_routh_array(coefs) do
    n = length(coefs)

    {row0, row1} =
      coefs
      |> Enum.with_index()
      |> Enum.split_with(fn {_, i} -> rem(i, 2) == 0 end)

    row0 = Enum.map(row0, fn {v, _} -> v end)
    row1 = Enum.map(row1, fn {v, _} -> v end)

    max_len = max(length(row0), length(row1))
    row0 = row0 ++ List.duplicate(0.0, max_len - length(row0))
    row1 = row1 ++ List.duplicate(0.0, max_len - length(row1))

    rows = [row0, row1]
    fill_routh_rows(rows, n - 2)
  end

  defp fill_routh_rows(rows, 0), do: rows

  defp fill_routh_rows([row_minus_2, row_minus_1 | _] = rows, remaining) do
    a = hd(row_minus_1)

    if abs(a) < 1.0e-15 do
      fill_routh_rows(rows ++ [List.duplicate(0.0, length(hd(rows)))], remaining - 1)
    else
      new_row =
        row_minus_1
        |> Enum.drop(1)
        |> Enum.with_index()
        |> Enum.map(fn {_, idx} ->
          b = Enum.at(row_minus_2, idx + 1) || 0.0
          c = Enum.at(row_minus_1, idx + 1) || 0.0
          -(Enum.at(row_minus_2, 0) * c - Enum.at(row_minus_1, 0) * b) / a
        end)

      pad_len = max(0, length(hd(rows)) - length(new_row))
      new_row = new_row ++ List.duplicate(0.0, pad_len)

      fill_routh_rows(rows ++ [new_row], remaining - 1)
    end
  end
end
