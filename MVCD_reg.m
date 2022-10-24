function [U, Z, V, B] = MVCD_reg(Adjmat, A, n, m, c, k, lmda, eta, gamma, verbose)
    U = rand(n, c);
    Z = rand(n, c);
    V = rand(m, k);
    B = tensor(rand(c, c, k), [c c k]);
    rowsum = 0.000000001 + Adjmat * ones(size(Adjmat, 2), 1);
    D = spdiags(rowsum(:), 0, n, n);
    iter = 0;
    maxIter = 2000;
    % last_obj = Inf;
    % epsil = eps;
    xi = eps;

    while (iter < maxIter)
        %% Update U
        P = double(tenmat(A, 1));
        % Q = double(tenmat(B, 1)) * kron(V, Z)';
        Q = ckronx({V, Z}, double(tenmat(B, 1))')';
        % R = double(tenmat(A, 1)) * kron(speye(m), Z')';
        R = ckronx({speye(m), Z'}, double(tenmat(A, 1))')';
        % S = speye(c) * double(tenmat(B, 1)) * kron(V, speye(c))';
        S = speye(c) * ckronx({V, speye(c)}, double(tenmat(B, 1))')';
        U = U .* ((P * Q' + lmda .* R * S' + eta .* Z + gamma .* Adjmat * U) ./ (xi + U * (Q * Q') + lmda .* R * R' * U + eta .* U + gamma .* D * U));
        %% Update Z
        P = double(tenmat(A, 2));
        % Q = double(tenmat(B, 2)) * kron(V, U)';
        Q = ckronx({V, U}, double(tenmat(B, 2))')';
        % R = double(tenmat(A, 2)) * kron(speye(m), U')';
        R = ckronx({speye(m), U'}, double(tenmat(A, 2))')';
        % S = speye(c) * double(tenmat(B, 2)) * kron(V, speye(c))';
        S = speye(c) * ckronx({V, speye(c)}, double(tenmat(B, 2))')';
        Z = Z .* ((P * Q' + lmda .* R * S' + eta .* U) ./ (xi + Z * (Q * Q') + lmda .* R * R' * Z + eta .* Z));
        %% Update V
        P = double(tenmat(A, 3));
        % Q = double(tenmat(B, 3)) * kron(Z, U)';
        % Q = prodABxC2(double(tenmat(B, 3)), Z', U');
        Q = ckronx({Z, U}, double(tenmat(B, 3))')';
        % R = speye(m) * double(tenmat(A, 3)) * kron(Z', U')';
        % R = speye(m) * prodABxC2(double(tenmat(A, 3)), Z, U);
        R = speye(m) * ckronx({Z', U'}, double(tenmat(A, 3))')';
        % S = double(tenmat(B, 3)) * kron(speye(c), speye(c))';
        S = double(tenmat(B, 3));
        V = V .* ((P * Q' + lmda .* R * S') ./ (xi + V * (Q * Q') + lmda .* V * (S * S')));
        %% Update B
        B = B .* (((1 + lmda) .* ttm(A, {U', Z', V'}, [1 2 3])) ./ (xi + ttm(B, {U' * U, Z' * Z, V' * V}, [1 2 3]) + lmda .* ttm(B, V' * V, 3)));
        
        % if verbose
        %     fprintf('This is %d-th iteration.\n', iter);
        % end

        if verbose
            obj = norm(A - ttm(B, {U, Z, V}, [1 2 3]))^2 + lmda .* norm(ttm(A, {U', Z'}, [1 2]) - ttm(B, V, 3))^2 + eta .* norm(U - Z, 'fro')^2 + gamma .* trace(U' * (D - Adjmat) * U);
            fprintf('This is %d th iteration, loss = %f.\n', iter + 1, obj);
        end

        % if (last_obj - obj) / last_obj < epsil
        %     break
        % end

        iter = iter + 1;
        % last_obj = obj;
    end

end
