defmodule NxLinAlg.Hessenberg do
  @moduledoc false

  alias NxLinAlg.Householder

  @doc """
  Reduces a square matrix A to upper Hessenberg form H = Q' * A * Q.

  Returns {H, Q}.
  """
  def hessenberg(a) do
    total = Nx.size(a)
    n = round(:math.sqrt(total))

    if n * n != total do
      raise ArgumentError, "hessenberg/1 requires a square matrix, got size #{total}"
    end

    a_f64 = Nx.as_type(a, :f64)
    q = Nx.eye(n, type: :f64)

    if n <= 2 do
      {a_f64, q}
    else
      # Apply Householder reflectors for columns 0 to n-3
      {h_final, q_final} =
        for k <- 0..(n - 3), reduce: {a_f64, q} do
          {a_acc, q_acc} ->
            # Extract column vector from position (k+1, k) downwards
            x = Nx.slice(a_acc, [k + 1, k], [n - k - 1, 1]) |> Nx.reshape({n - k - 1})

            # Compute Householder reflector
            {v, tau, _} = Householder.householder(x)

            if tau == 0.0 do
              {a_acc, q_acc}
            else
              # Build full reflector vector (padded with zeros above)
              padding = Nx.broadcast(0.0, {k + 1})
              v_full = Nx.concatenate([padding, v])  # length n

              # H_k = I - tau * v_full * v_full'
              h_mat = Nx.subtract(
                Nx.eye(n, type: :f64),
                Nx.multiply(tau, Nx.dot(Nx.reshape(v_full, {n, 1}), Nx.reshape(v_full, {1, n})))
              )

              # A = H_k * A * H_k
              ha = Nx.dot(h_mat, a_acc)
              a_new = Nx.dot(ha, h_mat)

              # Q = Q * H_k
              q_new = Nx.dot(q_acc, h_mat)

              {a_new, q_new}
            end
        end

      {h_final, q_final}
    end
  end
end
