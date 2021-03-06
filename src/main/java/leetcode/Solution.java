package leetcode;

import java.util.*;


/**
 * Stop adding more solutions here!
 */
class Solution {

    // 223
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int X0 = Math.max(A, E), Y0 = Math.max(B, F), X1 = Math.min(C, G), Y1 = Math.min(D, H);
        return area(A, B, C, D) + area(E, F, G, H) - area(X0, Y0, X1, Y1);
    }

    int area(int x0, int y0, int x1, int y1) {
        int w = x1 - x0, h = y1 - y0;
        if (w <= 0 || h <= 0) {
            return 0;
        }
        return w * h;
    }

    // 231
    boolean isPowerOfTwo(int n) {
        return (n > 0 && ((n & (n - 1)) == 0));
    }

    // 326
    boolean isPowerOfThree(int n) {
        return n > 0 && (1162261467 % n == 0);
    }

    // 153
    public int findMin(int[] nums) {
        int low = 0;
        int high = nums.length - 1;
        while (low < high) {
            int pivot = low + (high - low) / 2;
            if (nums[pivot] < nums[high]) {
                high = pivot;
            } else {
                low = pivot + 1;
            }
        }
        return nums[low];
    }

    public boolean searchV2(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return false;
        }
        if (n == 1) {
            return nums[0] == target;
        }
        int l = 0, r = n - 1, mid;
        while (l <= r) {
            mid = (l + r) / 2;
            if (nums[mid] == target) {
                return true;
            }
            if (nums[l] == nums[mid] && nums[mid] == nums[r]) {
                ++l;
                --r;
            } else if (nums[l] <= nums[mid]) {
                if (nums[l] <= target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return false;
    }

    // 33
    public int search(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return -1;
        }
        if (n == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int low = 0, high = nums.length - 1, mid;
        while (low <= high) {

            mid = low + (high - low) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > nums[low]) { // ?????????????????????????????????
                if (nums[low] <= target && target < nums[mid]) {
                    high = mid - 1;
                } else {
                    low = mid + 1;
                }
            } else if (nums[mid] < nums[high]) { // ?????????????????????????????????
                if (nums[mid] < target && target <= nums[high]) {
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            } else {
                if (nums[low] == nums[mid]) {
                    low++;
                }
                if (nums[high] == nums[mid]) {
                    high--;
                }
            }
        }
        return -1;
    }

    // 50
    public double myPow(double x, int n) {
        return (long) n >= 0 ? quickMul(x, n) : 1.0 / quickMul(x, -(long) n);
    }

    double quickMul(double x, long N) {
        double ans = 1.0;
        double x_contribute = x;
        while (N > 0) {
            if (N % 2 == 1) {
                ans *= x_contribute;
            }
            x_contribute *= x_contribute;
            N /= 2;
        }
        return ans;
    }

    // 80
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        if (n <= 2) {
            return n;
        }
        int slow = 2, fast = 2;
        while (fast < n) {
            if (nums[slow - 2] != nums[fast]) {
                nums[slow] = nums[fast];
                slow += 1;
            }
            fast += 1;
        }
        return slow;
    }

    // 88
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = m - 1, p2 = n - 1;
        int tail = m + n - 1;
        int cur;
        while (p1 >= 0 || p2 >= 0) {
            if (p1 == -1) {
                cur = nums2[p2--];
            } else if (p2 == -1) {
                cur = nums1[p1--];
            } else if (nums1[p1] > nums2[p2]) {
                cur = nums1[p1--];
            } else {
                cur = nums2[p2--];
            }
            nums1[tail--] = cur;
        }
    }

    // 781
    public int numRabbits(int[] answers) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int y : answers) {
            count.put(y, count.getOrDefault(y, 0) + 1);
        }
        int ans = 0;
        for (Map.Entry<Integer, Integer> entry : count.entrySet()) {
            int y = entry.getKey(), x = entry.getValue();
            ans += (x + y) / (y + 1) * (y + 1);
        }
        return ans;
    }

    // ????????? 17.21. ??????????????????
    public int trap(int[] height) {
        int n = height.length;
        if (n == 0) {
            return 0;
        }

        int[] leftMax = new int[n];
        leftMax[0] = height[0];
        for (int i = 1; i < n; ++i) {
            leftMax[i] = Math.max(leftMax[i - 1], height[i]);
        }

        int[] rightMax = new int[n];
        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            rightMax[i] = Math.max(rightMax[i + 1], height[i]);
        }

        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans += Math.min(leftMax[i], rightMax[i]) - height[i];
        }
        return ans;
    }

    // 1006
    public int clumsy(int N) {
        Deque<Integer> stack = new LinkedList<>();
        stack.push(N);
        N--;

        int index = 0;
        while (N > 0) {
            if (index % 4 == 0) {
                stack.push(stack.pop() * N);
            } else if (index % 4 == 1) {
                stack.push(stack.pop() / N);
            } else if (index % 4 == 2) {
                stack.push(N);
            } else {
                stack.push(-N);
            }
            index++;
            N--;
        }

        int sum = 0;
        while (!stack.isEmpty()) {
            sum += stack.pop();
        }
        return sum;
    }

    public int largestAltitude(int[] gain) {
        int res = 0, t = 0;
        for (int x : gain) {
            t += x;
            res = Math.max(res, t);
        }
        return res;
    }

    // 74
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = matrix.length, col = matrix[0].length;
        int low = 0, high = row * col - 1;
        while (low <= high) {
            int mid = (high - low) / 2 + low;
            int x = matrix[mid / col][mid % col];
            if (x < target) {
                low = mid + 1;
            } else if (x > target) {
                high = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }

    // 202
    public boolean isHappy(int n) {
        // ????????????
        int slow = n, fast = squareSum(n);
        while (slow != fast) {
            slow = squareSum(slow);
            fast = squareSum(squareSum(fast));
        }
        return slow == 1;
    }

    int squareSum(int n) {
        int sum = 0;
        while (n > 0) {
            int digit = n % 10;
            sum += digit * digit;
            n /= 10;
        }
        return sum;
    }

    // 190
    public int reverseBits(int n) {

        long res = 0;
        for (int i = 0; i < 32; i++) {
            res = (res << 1) | (n & 1);
            n >>>= 1;
        }
        return (int) res;
    }

    // 119
    public List<Integer> getRow(int rowIndex) {
        var row = new ArrayList<Integer>();
        row.add(1);
        for (int i = 1; i <= rowIndex; ++i) {
            row.add((int) ((long) row.get(i - 1) * (rowIndex - i + 1) / i));
        }
        return row;
    }

    // 118
    public List<List<Integer>> generate(int numRows) {
        var ret = new ArrayList<List<Integer>>();
        for (int i = 0; i < numRows; ++i) {
            var row = new ArrayList<Integer>();
            for (int j = 0; j <= i; ++j) {
                if (j == 0 || j == i) {
                    row.add(1);
                } else {
                    row.add(ret.get(i - 1).get(j - 1) + ret.get(i - 1).get(j));
                }
            }
            ret.add(row);
        }
        return ret;
    }

    // 58
    public int lengthOfLastWord(String s) {
        int r = s.length() - 1;
        while (r >= 0 && s.charAt(r) == ' ') {
            r--;
        }
        if (r < 0) {
            return 0;
        }
        int l = r;
        while (l >= 0 && s.charAt(l) != ' ') {
            l--;
        }
        return r - l;
    }

    // 26 ????????????
    public int removeDuplicatesIntArray(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int i = 0; // slow
        for (int j = 1; j < nums.length; j++) {
            if (nums[j] != nums[i]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }

    public int numDistinct(String s, String t) {
        int m = s.length(), n = t.length();
        if (m < n) {
            return 0;
        }
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i < m + 1; i++) {
            dp[i][n] = 1;
        }
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                if (s.charAt(i) == t.charAt(j)) {
                    dp[i][j] = dp[i + 1][j + 1] + dp[i + 1][j];
                } else {
                    dp[i][j] = dp[i + 1][j];
                }
            }
        }
        return dp[0][0];
    }

    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        if (n == 1) {
            res[0][0] = 1;
            return res;
        }

        int maxNum = n * n;
        int curNum = 1;
        int row = 0, col = 0;
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}; // ????????????
        int directionIndex = 0;
        while (curNum <= maxNum) {
            res[row][col] = curNum;
            curNum++;
            int nextRow = row + directions[directionIndex][0];
            int nextCol = col + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= n || nextCol < 0 || nextCol >= n
                    || res[nextRow][nextCol] != 0) {
                directionIndex = (directionIndex + 1) % 4;
            }
            row = row + directions[directionIndex][0];
            col = col + directions[directionIndex][1];
        }
        return res;
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        var res = new LinkedList<Integer>();

        var row = matrix.length;
        if (row == 0) {
            return res;
        }

        var col = matrix[0].length;
        if (col == 0) {
            return res;
        }

        // top???left???right???bottom ??????????????????????????????????????????????????????
        int top = 0, left = 0, bottom = row - 1, right = col - 1;
        int count = 0, sum = row * col;

        // ??????????????????????????????
        while (count < sum) {
            int i = top, j = left;
            while (j <= right && count < sum) {
                res.addLast(matrix[i][j]);
                count++;
                j++;
            }
            i = top + 1;
            j = right;
            while (i <= bottom && count < sum) {
                res.addLast(matrix[i][j]);
                count++;
                i++;
            }
            i = bottom;
            j = right - 1;
            while (j >= left && count < sum) {
                res.addLast(matrix[i][j]);
                count++;
                j--;
            }
            i = bottom - 1;
            j = left;

            while (i > top && count < sum) {
                res.addLast(matrix[i][j]);
                count++;
                i--;
            }
            // ??????????????????
            top = top + 1;
            left = left + 1;
            bottom = bottom - 1;
            right = right - 1;
        }
        return res;

    }

    public boolean isValidSerialization(String preorder) {

        var nodes = preorder.split(",");
        int diff = 1;
        for (String node : nodes) {
            diff--;
            if (diff < 0) {
                return false;
            }
            if (!node.equals("#")) {
                diff += 2;
            }

        }

        return diff == 0;
    }

    public int[] plusOne(int[] digits) {
        for (int i = digits.length - 1; i >= 0; i--) {
            digits[i]++;
            digits[i] = digits[i] % 10;
            if (digits[i] != 0) {
                return digits;
            }
        }
        digits = new int[digits.length + 1];
        digits[0] = 1;
        return digits;
    }

    public int calculate(String s) {
        var stack = new Stack<Integer>();
        char preSign = '+';
        int num = 0, n = s.length();
        for (int i = 0; i < n; ++i) {
            if (Character.isDigit(s.charAt(i))) {
                num = num * 10 + s.charAt(i) - '0';
            }
            if (!Character.isDigit(s.charAt(i)) && s.charAt(i) != ' ' || i == n - 1) {
                switch (preSign) {
                    case '+':
                        stack.push(num);
                        break;
                    case '-':
                        stack.push(-num);
                        break;
                    case '*':
                        stack.push(stack.pop() * num);
                        break;
                    default:
                        stack.push(stack.pop() / num);
                }
                preSign = s.charAt(i);
                num = 0;

            }
        }
        int res = 0;
        for (Integer i : stack) {
            res += i;
        }
        return res;
    }

    public String removeDuplicatesStr(String S) {
        Stack<Character> stack = new Stack<>();
        char[] cs = S.toCharArray();
        for (char c : cs) {
            if (stack.empty() || c != stack.peek()) {
                stack.push(c);
            } else {
                stack.pop();
            }
        }
        StringBuilder res = new StringBuilder();
        for (Character c : stack) {
            res.append(c);
        }
        return res.toString();
    }

    public int minCut(String s) {
        int n = s.length();
        boolean[][] g = new boolean[n][n];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(g[i], true);
        }

        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                g[i][j] = s.charAt(i) == s.charAt(j) && g[i + 1][j - 1];
            }
        }

        int[] f = new int[n];
        Arrays.fill(f, Integer.MAX_VALUE);
        for (int i = 0; i < n; ++i) {
            if (g[0][i]) {
                f[i] = 0;
            } else {
                for (int j = 0; j < i; ++j) {
                    if (g[j + 1][i]) {
                        f[i] = Math.min(f[i], f[j] + 1);
                    }
                }
            }
        }

        return f[n - 1];
    }


    // 7 ????????????
    public int reverse(int x) {

        long res = 0;

        while (x != 0) {
            res = res * 10 + x % 10;
            x /= 10;
        }
        return (int) res == res ? (int) res : 0;
    }

    public boolean find132pattern(int[] nums) {
        if (nums.length < 3) {
            return false;
        }
        var num3 = Integer.MIN_VALUE;
        var stack = new Stack<Integer>();

        for (int i = nums.length - 1; i >= 0; i--) {
            if (nums[i] < num3) {
                return true;
            }
            while (stack.size() != 0 && nums[i] > stack.peek()) {
                num3 = stack.pop();
            }
            stack.push(nums[i]);
        }
        return false;
    }

    public int hammingDistance(int x, int y) {
        int xor = x ^ y, distance = 0;
        while (xor > 0) {
            if ((xor & 1) > 0) {
                distance++;
            }
            xor >>= 1;
        }
        return distance;
    }

    public void moveZeroes(int[] nums) {
        int n = nums.length, leftTail = 0, rightHead = 0, temp;
        while (rightHead < n) {
            if (nums[rightHead] != 0) {
                temp = nums[leftTail];
                nums[leftTail] = nums[rightHead];
                nums[rightHead] = temp;
                leftTail++;
            }
            rightHead++;
        }
    }

    // 191 ??? 1 ?????????
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            n = n & (n - 1);
            count += 1;
        }
        return count;
    }

    // 69 ????????????
    public int mySqrt(int x) {
        int l = 0, r = x, ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if ((long) mid * mid <= x) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }

    // 78
    public List<List<Integer>> subsets(int[] nums) {
        var ans = new ArrayList<List<Integer>>();
        var aux = new ArrayList<Integer>();

        int n = nums.length;
        for (int mask = 0; mask < (1 << n); ++mask) {
            aux.clear();
            for (int i = 0; i < n; ++i) {
                if ((mask & (1 << i)) != 0) {
                    aux.add(nums[i]);
                }
            }
            ans.add(new ArrayList<>(aux));
        }
        return ans;
    }

    /**
     * @param haystack source
     * @param needle   target
     * @return ??? haystack ?????????????????? needle ????????????????????????????????????????????? 0 ?????????????????????????????? -1
     */
    public int strStr(String haystack, String needle) {
        int lenH = haystack.length(), lenN = needle.length();
        if (lenN == 0) {
            return 0;
        }
        // ?????????
        int cursorH = 0;
        while (cursorH < lenH - lenN + 1) {
            // ??????????????? ?????? needle ??????????????????
            while (cursorH < lenH - lenN + 1 && haystack.charAt(cursorH) != needle.charAt(0)) {
                cursorH++;
            }
            int curMatch = 0, cursorN = 0;
            // ?????? ??????
            while (cursorN < lenN && cursorH < lenH && haystack.charAt(cursorH) == needle
                    .charAt(cursorN)) {
                cursorH++;
                cursorN++;
                curMatch++;
            }
            // ????????????
            if (curMatch == lenN) {
                return cursorH - lenN;
            }
            // ??????
            cursorH = cursorH - curMatch + 1;
        }
        return -1;

    }

    // 219
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; ++i) {
            if (set.contains(nums[i])) {
                return true;
            }
            set.add(nums[i]);
            if (set.size() > k) {
                set.remove(nums[i - k]);
            }
        }
        return false;
    }

    // 217
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (!set.add(num)) {
                return true;
            }
        }
        return false;
    }

    // 53
    public int maxSubArray(int[] nums) {
        int pre = 0, maxAns = nums[0];
        for (int x : nums) {
            pre = Math.max(pre + x, x);
            maxAns = Math.max(maxAns, pre);
        }
        return maxAns;
    }

    // 87
    public boolean isScramble(String s1, String s2) {
        if (s1.length() != s2.length()) {
            return false;
        }
        if (s1.equals(s2)) {
            return true;
        }

        int[] letterCount = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            letterCount[s1.charAt(i) - 'a']++;
            letterCount[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++) {
            if (letterCount[i] != 0) {
                return false;
            }
        }

        int length = s1.length();
        boolean[][][] dp = new boolean[length + 1][length][length];
        // ??????????????????????????????
        for (int len = 1; len <= length; len++) {
            // S1 ???????????????
            for (int i = 0; i + len <= length; i++) {
                // S2 ???????????????
                for (int j = 0; j + len <= length; j++) {
                    // ????????? 1 ????????????
                    if (len == 1) {
                        dp[len][i][j] = s1.charAt(i) == s2.charAt(j);
                    } else {
                        // ????????????????????????????????????
                        for (int q = 1; q < len; q++) {
                            dp[len][i][j] = dp[q][i][j] && dp[len - q][i + q][j + q]
                                    || dp[q][i][j + len - q] && dp[len - q][i + q][j];
                            // ??????????????? true ??? break????????????????????? false
                            if (dp[len][i][j]) {
                                break;
                            }
                        }
                    }
                }
            }
        }
        return dp[length][0][0];
    }

    // 35
    public int searchInsert(int[] nums, int target) {

        int l = 0, r = nums.length, mid;
        while (l < r) {
            mid = l + (r - l) / 2;
            if (target == nums[mid]) {
                return mid;
            } else if (target < nums[mid]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    // 11
    public int maxArea(int[] height) {
        int l = 0, r = height.length - 1;
        int ans = 0;
        while (l < r) {
            int area = Math.min(height[l], height[r]) * (r - l);
            ans = Math.max(ans, area);
            if (height[l] <= height[r]) {
                l += 1;
            } else {
                r -= 1;
            }
        }
        return ans;
    }

}
