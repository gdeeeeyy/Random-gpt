import java.util.*;

class Python {
    public static void main(String[] args) {
        int n = 4;
        Integer[] ex = new Integer[n];
        int nex = 0;
        Integer[] sx = new Integer[n];
        int nsx = 0;
        int piv = -1;
        for (int i = 0; i < n; i++) {
            nsx += (i + 1);
            nex += (n - (i + 1));
            sx[i] = nsx;
            ex[i] = nex;
        }
        List<Integer> startArr = Arrays.asList(sx);
        List<Integer> endArr = Arrays.asList(ex);
        System.out.println(startArr);
        System.out.println(endArr);
        for (int i = 0; i < n; i++) {
            for (int j = n - 1; j >= 1; j--) {
                if (ex[i] == sx[j]) {
                    piv = i - 1;
                }
            }
        }
        System.out.println(piv);
    }
}