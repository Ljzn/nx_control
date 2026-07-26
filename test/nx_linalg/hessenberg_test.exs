defmodule NxLinAlg.HessenbergTest do
  use ExUnit.Case, async: true

  alias NxLinAlg.Hessenberg

  def assert_all_close(a, b, tol \\ 1.0e-6) do
    diff = Nx.subtract(Nx.as_type(a, :f64), Nx.as_type(b, :f64))
    max_err = Nx.to_number(Nx.reduce_max(Nx.abs(diff)))
    assert max_err < tol, "max diff #{max_err} >= #{tol}"
  end

  describe "hessenberg/1" do
    test "3x3 matrix reconstruction" do
      a = Nx.tensor([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]], type: :f64)
      {h, q} = Hessenberg.hessenberg(a)

      # A ≈ Q * H * Q'
      assert_all_close(a, Nx.dot(Nx.dot(q, h), Nx.transpose(q)), 1.0e-6)
    end

    test "4x4 matrix reconstruction" do
      # Non-singular matrix (Hilbert-like)
      a = Nx.tensor([[4.0, 1.0, 0.0, 0.0],
                     [1.0, 4.0, 1.0, 0.0],
                     [0.0, 1.0, 4.0, 1.0],
                     [0.0, 0.0, 1.0, 4.0]], type: :f64)
      {h, q} = Hessenberg.hessenberg(a)

      assert_all_close(a, Nx.dot(Nx.dot(q, h), Nx.transpose(q)), 1.0e-6)
    end

    test "tridiagonal 5x5 reconstruction" do
      a = Nx.tensor([[2.0, -1.0, 0.0, 0.0, 0.0],
                     [-1.0, 2.0, -1.0, 0.0, 0.0],
                     [0.0, -1.0, 2.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0, 2.0, -1.0],
                     [0.0, 0.0, 0.0, -1.0, 2.0]], type: :f64)
      {h, q} = Hessenberg.hessenberg(a)

      assert_all_close(a, Nx.dot(Nx.dot(q, h), Nx.transpose(q)), 1.0e-6)
    end
  end
end
