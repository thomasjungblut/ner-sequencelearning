package de.jungblut.conll;

import java.util.Map;
import java.util.Map.Entry;

import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

public class LabelManager {

    private final BiMap<String, Integer> labelMap = HashBiMap.create();
    private boolean forbidAdditions = false;
    private int nextLabel = 0;

    public LabelManager() {
        forbidAdditions = false;
    }

    public LabelManager(Map<Integer, String> existingLabels) {
        forbidAdditions = true;

        System.out.println("loading existing labels: " + existingLabels);
        for (Entry<Integer, String> entry : existingLabels.entrySet()) {
            labelMap.put(entry.getValue(), entry.getKey());
        }
    }

    public int getOrCreate(String key) {
        Integer index = labelMap.get(key);
        if (index == null) {
            Preconditions.checkState(!forbidAdditions,
                    "can't add labels while additions to the labels are forbidden");
            index = nextLabel++;
            labelMap.put(key, index);
        }
        return index;
    }

    public BiMap<String, Integer> getLabelMap() {
        return labelMap;
    }

}
