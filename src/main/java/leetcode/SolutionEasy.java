package leetcode;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class SolutionEasy {

  // 121
  public int maxProfit(int[] prices) {
    int minPrice = Integer.MAX_VALUE;
    int maxProfit = 0;
    for (int price : prices) {
      if (price < minPrice) {
        minPrice = price;
      } else if (price - minPrice > maxProfit) {
        maxProfit = price - minPrice;
      }
    }
    return maxProfit;
  }

  // 263
  public boolean isUgly(int num) {

    if (num <= 0) {
      return false;
    }
    int[] primes = {2, 3, 5};
    for (int prime : primes) {
      while (num % prime == 0) {
        num /= prime;
      }
    }
    return num == 1;

  }

  // 27
  public int removeElement(int[] nums, int val) {
    int l = 0, r = nums.length;
    while (l < r) {
      if (nums[l] == val) {
        nums[l] = nums[r - 1];
        r--;
      } else {
        l++;
      }
    }
    return l;
  }

  // 1576
  public String modifyString(String s) {
    var chars = s.toCharArray();
    int n = chars.length;
    for (int i = 0; i < chars.length; i++) {
      if (chars[i] == '?') {
        char left = i == 0 ? 'S' : chars[i - 1];
        char right = i == n - 1 ? 'S' : chars[i + 1];
        char temp = 'a';
        while (temp == left || temp == right) {
          temp++;
        }
        chars[i] = temp;
      }
    }
    return new String(chars);
  }

  // 1207
  public boolean uniqueOccurrences(int[] arr) {
    Map<Integer, Integer> times = new HashMap<>();
    for (int x : arr) {
      times.put(x, times.getOrDefault(x, 0) + 1);
    }
    Set<Integer> deduplication = new HashSet<>();
    for (Map.Entry<Integer, Integer> x : times.entrySet()) {
      if (!deduplication.add(x.getValue())) {
        return false;
      }
    }
    return true;
  }

  // 136
  public int singleNumber(int[] nums) {
    int ans = 0;
    for (var num : nums) {
      ans ^= num;
    }
    return ans;
  }

}
