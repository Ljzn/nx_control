defmodule NxControl.TransferFunctionTest do
  use ExUnit.Case, async: true

  describe "new/2" do
    test "creates a transfer function from lists" do
      tf = NxControl.TransferFunction.new([1], [1, 2, 1])
      assert Nx.to_flat_list(tf.num) == [1.0]
      assert Nx.to_flat_list(tf.den) == [1.0, 2.0, 1.0]
    end
  end

  describe "stable?/1" do
    test "stable second-order system" do
      tf = NxControl.TransferFunction.new([1], [1, 3, 2])
      assert NxControl.TransferFunction.stable?(tf)
    end

    test "unstable system with negative coefficient" do
      tf = NxControl.TransferFunction.new([1], [1, -1, -6])
      refute NxControl.TransferFunction.stable?(tf)
    end

    test "stable fourth-order system" do
      tf = NxControl.TransferFunction.new([1], [1, 10, 35, 50, 24])
      assert NxControl.TransferFunction.stable?(tf)
    end

    test "unstable system with missing term" do
      tf = NxControl.TransferFunction.new([1], [1, 0, 1, 1])
      refute NxControl.TransferFunction.stable?(tf)
    end
  end
end
