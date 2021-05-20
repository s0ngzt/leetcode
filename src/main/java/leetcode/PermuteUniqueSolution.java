package leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class PermuteUniqueSolution {

  public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> perm = new ArrayList<>();
    boolean[] visited = new boolean[nums.length];
    Arrays.sort(nums);
    backtrack(nums, visited, perm, res);
    return res;
  }

  public void backtrack(int[] nums, boolean[] visited, List<Integer> perm,
      List<List<Integer>> res) {
    // 满足结束条件
    if (perm.size() == nums.length) {
      res.add(new ArrayList<>(perm));
      return;
    }
    for (int i = 0; i < nums.length; i++) {
      // 已经添加过的元素 或 上一个元素和当前相同，且没有访问过
      if (visited[i] || (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1])) {
        continue;
      }
      // 做选择
      perm.add(nums[i]);
      visited[i] = true;
      backtrack(nums, visited, perm, res);
      // 撤销选择
      visited[i] = false;
      perm.remove(perm.size() - 1);
    }
  }
}
