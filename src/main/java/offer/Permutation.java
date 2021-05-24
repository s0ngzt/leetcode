package offer;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

// 38. 字符串的排列
class Permutation {

  List<String> res = new LinkedList<>();
  char[] characters;

  public String[] permutation(String s) {
    characters = s.toCharArray();
    dfs(0);
    return res.toArray(new String[res.size()]);
  }

  void dfs(int x) {
    if (x == characters.length - 1) {
      // 添加排列方案
      res.add(String.valueOf(characters));
      return;
    }
    Set<Character> set = new HashSet<>();
    for (int i = x; i < characters.length; i++) {
      if (set.contains(characters[i])) {
        continue; // 剪枝
      }
      set.add(characters[i]);
      swap(i, x); // 交换，将 c[i] 固定在第 x 位
      dfs(x + 1); // 开启固定第 x + 1 位字符
      swap(i, x); // 恢复交换
    }
  }

  void swap(int i, int j) {
    char tmp = characters[i];
    characters[i] = characters[j];
    characters[j] = tmp;
  }
}
