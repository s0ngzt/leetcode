package interview;

import java.util.HashMap;
import java.util.Map;

// 面试题 16.02 - 单词频率
class WordsFrequency {

    Map<String, Integer> count;

    public WordsFrequency(String[] book) {
        count = new HashMap<>();
        for (var word : book) {
            count.put(word, count.getOrDefault(word, 0) + 1);
        }
    }

    public int get(String word) {
        return count.getOrDefault(word, 0);
    }

}
