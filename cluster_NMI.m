function [nmi] = cluster_NMI(labels_true, labels_pred)
    labels_true = labels_true(:)';
    labels_pred = labels_pred(:)';
    n = length(labels_true);

    Hsigma = 0;
    %% iterate over all detected communities.
    for i1 = unique(labels_pred)
        freq = length(find(labels_pred == i1)) / n;
        Hsigma = Hsigma - (freq * log(freq));
    end

    Hc = 0;
    %% iterate over all detected communities.
    for i2 = unique(labels_true)
        freq = length(find(labels_true == i2)) / n;
        Hc = Hc - (freq * log(freq));
    end

    I = 0;

    for i1 = unique(labels_pred)

        for i2 = unique(labels_true)
            intersection = length(intersect(find(labels_pred == i1), find(labels_true == i2)));
            prodct = length(find(labels_pred == i1)) * length(find(labels_true == i2));

            if intersection ~= 0
                I = I + (intersection / n) * log(n * intersection / prodct);
            end

        end

    end

    nmi = 2.0 * I / (Hsigma + Hc);
end
