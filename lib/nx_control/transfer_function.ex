defmodule NxControl.TransferFunction do
  @moduledoc """
  Transfer function representation G(s) = num(s) / den(s).
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

    # All coefficients must be the same sign (positive for standard form)
    sign = if hd(den) > 0, do: 1, else: -1

    if Enum.any?(den, fn c -> c * sign <= 0 end) do
      false
    else
      routh_array = build_routh_array(den)
      first_col = Enum.map(routh_array, &List.first/1)
      Enum.all?(first_col, &(&1 * sign > 0))
    end
  end

  # Builds the Routh array from denominator coefficients.
  # Returns a list of rows, each row is a list of numbers.
  defp build_routh_array(coefs) do
    n = length(coefs)
    num_rows = n

    # First two rows: even-indexed and odd-indexed coefficients
    {row0, row1} =
      coefs
      |> Enum.with_index()
      |> Enum.split_with(fn {_, i} -> rem(i, 2) == 0 end)

    row0 = Enum.map(row0, fn {v, _} -> v end)
    row1 = Enum.map(row1, fn {v, _} -> v end)

    # Pad rows to equal length (pad with zeros on the right)
    max_len = max(length(row0), length(row1))
    row0 = row0 ++ List.duplicate(0.0, max_len - length(row0))
    row1 = row1 ++ List.duplicate(0.0, max_len - length(row1))

    rows = [row0, row1]
    fill_routh_rows(rows, num_rows - 2)
  end

  defp fill_routh_rows(rows, 0), do: rows

  defp fill_routh_rows([row_minus_2, row_minus_1 | _] = rows, remaining) do
    a = hd(row_minus_1)

    if abs(a) < 1.0e-15 do
      # Zero in first column → degenerate Routh array → unstable
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
