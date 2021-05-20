package leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class SubsetIISolution {

  List<Integer> l;
  List<List<Integer>> res;

  public List<List<Integer>> subsetsWithDup(int[] nums) {
    l = new ArrayList<>();
    res = new ArrayList<>();
    Arrays.sort(nums); // 排序
    subsetsWithDup(nums, 0, false);
    return res;
  }

  public void subsetsWithDup(int[] nums, int i, boolean choosePre) {
    if (i == nums.length) {
      res.add(new ArrayList<>(l));
      return;
    }

    l.add(nums[i]);
    subsetsWithDup(nums, i + 1, true);
    l.remove(l.size() - 1);

    if (choosePre && nums[i - 1] == nums[i]) {
      return;
    }
    subsetsWithDup(nums, i + 1, false);
  }

}
