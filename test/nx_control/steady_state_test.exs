defmodule NxControl.SteadyStateTest do
  use ExUnit.Case, async: true
  doctest NxControl.SteadyState

  alias NxControl.{TransferFunction, SteadyState}

  describe "system_type/1" do
    test "type 0 system (no integrators)" do
      tf = TransferFunction.new([1], [1, 3, 2])
      assert SteadyState.system_type(tf) == 0
    end

    test "type 1 system (one integrator)" do
      tf = TransferFunction.new([1], [1, 1, 0])
      assert SteadyState.system_type(tf) == 1
    end

    test "type 2 system (two integrators)" do
      tf = TransferFunction.new([1], [1, 1, 0, 0])
      assert SteadyState.system_type(tf) == 2
    end
  end

  describe "step_error/1" do
    test "type 0 has finite error" do
      tf = NxControl.TransferFunction.new([1], [1, 1])
      assert_in_delta SteadyState.step_error(tf), 0.5, 1.0e-12
    end

    test "type 1 has zero error" do
      tf = TransferFunction.new([5], [1, 1, 0])
      assert SteadyState.step_error(tf) == 0.0
    end
  end

  describe "ramp_error/1" do
    test "type 0 has infinite error" do
      tf = TransferFunction.new([1], [1, 3, 2])
      assert SteadyState.ramp_error(tf) == :infinity
    end

    test "type 1 has finite error = 1/Kv" do
      tf = TransferFunction.new([5], [1, 1, 0])
      assert_in_delta SteadyState.ramp_error(tf), 0.2, 1.0e-12
    end
  end

  describe "parabolic_error/1" do
    test "type 0 has infinite error" do
      tf = TransferFunction.new([1], [1, 3, 2])
      assert SteadyState.parabolic_error(tf) == :infinity
    end

    test "type 2 has finite error = 1/Ka" do
      tf = TransferFunction.new([5], [1, 1, 0, 0])
      assert_in_delta SteadyState.parabolic_error(tf), 0.2, 1.0e-12
    end
  end
end
