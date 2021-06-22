package algo;

class BinarySearch {

    /**
     * 二分查找
     *
     * @param nums 数组
     * @param target 目标
     * @return 下标，不存在返回 -1
     */
    static int binarySearch(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return -1;
        }
        int l = 0, r = n, mid;
        while (l < r) {
            mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (target < nums[mid]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return -1;
    }

    static int leftBound(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return -1;
        }
        int l = 0, r = n, mid;
        while (l < r) {
            mid = l + (r - l) / 2;
            if (target <= nums[mid]) {
                // 找到 target 时缩小上界
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        if (l == n || nums[l] != target) {
            return -1;
        }
        return l;
    }

    static int rightBound(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return -1;
        }
        int l = 0, r = n, mid;
        while (l < r) {
            mid = l + (r - l) / 2;
            if (nums[mid] <= target) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        if (r > 0 && nums[r - 1] != target) {
            return -1;
        }
        return r - 1;
    }
}
