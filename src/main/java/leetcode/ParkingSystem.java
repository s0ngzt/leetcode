package leetcode;

class ParkingSystem {

    int[] nums;

    public ParkingSystem(int big, int medium, int small) {
        nums = new int[3];
        nums[0] = big;
        nums[1] = medium;
        nums[2] = small;
    }

    public boolean addCar(int carType) {
        if (nums[carType - 1] > 0) {
            nums[carType - 1] -= 1;
            return true;
        }
        return false;
    }
}
