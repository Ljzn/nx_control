defmodule NxLinAlg.HouseholderTest do
  use ExUnit.Case, async: true

  alias NxLinAlg.Householder

  def assert_all_close(a, b, tol \\ 1.0e-6) do
    diff = Nx.subtract(Nx.as_type(a, :f64), Nx.as_type(b, :f64))
    max_err = Nx.to_number(Nx.reduce_max(Nx.abs(diff)))
    assert max_err < tol, "max diff #{max_err} >= #{tol}"
  end

  describe "householder/1" do
    test "reflects x to ±||x||·e₁" do
      x = Nx.tensor([3.0, 1.0, 2.0], type: :f64)
      {v, tau, beta} = Householder.householder(x)

      n = Nx.size(x)
      h = Nx.subtract(
        Nx.eye(n, type: :f64),
        Nx.multiply(tau, Nx.dot(Nx.reshape(v, {n, 1}), Nx.reshape(v, {1, n})))
      )
      hx = Nx.dot(h, x)

      beta_t = Nx.tensor(beta, type: :f64) |> Nx.reshape({1})
      assert_all_close(hx[0..0] |> Nx.flatten(), beta_t)
      tail_abs = Nx.abs(hx[1..-1//1])
      assert Nx.to_number(Nx.reduce_max(tail_abs)) < 1.0e-6
    end
  end

  describe "apply_left/3" do
    test "applies reflector to matrix" do
      a = Nx.tensor([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]], type: :f64)
      x = Nx.flatten(a[0..0])
      {v, tau, _} = Householder.householder(x)

      result = Householder.apply_left(v, tau, a)
      assert Nx.shape(result) == {3, 3}
    end
  end

  describe "apply_right/3" do
    test "applies reflector to matrix" do
      a = Nx.tensor([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]], type: :f64)
      x = Nx.flatten(Nx.transpose(a)[0..0])
      {v, tau, _} = Householder.householder(x)

      result = Householder.apply_right(v, tau, a)
      assert Nx.shape(result) == {3, 3}
    end
  end
end
