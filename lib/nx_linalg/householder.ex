defmodule NxLinAlg.Householder do
  @moduledoc false

  @doc """
  Standard Householder reflector.

  Given vector x, returns {v, tau, beta} such that H = I - tau*v*v'
  reflects x to [beta, 0, ..., 0]^T with beta = -sign(x[0])*||x||.

  The reflector vector v is normalized so that v[0] = 1.
  """
  def householder(x) do
    n = Nx.size(x)
    x0 = Nx.to_number(Nx.reshape(x[0..0], {}))

    if n == 1 do
      {Nx.tensor([1.0]), 0.0, x0}
    else
      tail = x[1..-1//1]
      sigma = Nx.sum(Nx.pow(tail, 2)) |> Nx.to_number()

      if sigma < 1.0e-300 do
        v = Nx.concatenate([Nx.tensor([1.0]), Nx.broadcast(0.0, {n - 1})])
        {v, 0.0, x0}
      else
        norm = :math.sqrt(x0 * x0 + sigma)
        beta = if x0 < 0, do: norm, else: -norm

        u0 = x0 - beta
        v_tail = Nx.divide(tail, u0)
        v = Nx.concatenate([Nx.tensor([1.0]), v_tail])

        v_norm = Nx.sum(Nx.pow(v_tail, 2)) |> Nx.to_number()
        tau = 2.0 / (1.0 + v_norm)

        {v, tau, beta}
      end
    end
  end

  @doc """
  Applies H = I - tau*v*v' to the left of A: returns H*A.
  """
  def apply_left(v, tau, a) do
    n = Nx.size(v)
    m = div(Nx.size(a), n)
    a2 = Nx.reshape(a, {m, n})

    vt = Nx.new_axis(v, 0)
    v_c = Nx.new_axis(v, 1)
    vta = Nx.dot(vt, a2)
    correction = Nx.dot(v_c, Nx.multiply(tau, vta))
    Nx.subtract(a2, correction)
  end

  @doc """
  Applies H = I - tau*v*v' to the right of A: returns A*H.
  """
  def apply_right(v, tau, a) do
    n = Nx.size(v)
    m = div(Nx.size(a), n)
    a2 = Nx.reshape(a, {m, n})

    v_c = Nx.new_axis(v, 1)
    vt = Nx.new_axis(v, 0)
    av = Nx.dot(a2, v_c)
    correction = Nx.dot(Nx.multiply(tau, av), vt)
    Nx.subtract(a2, correction)
  end
end
