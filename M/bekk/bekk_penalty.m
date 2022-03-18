function penalty = bekk_penalty(C,A,B,lambdaDiagonal,matricesDiagonal,regularizationDiagonal,lambdaFull,matricesFull,regularizationFull)

if isempty(matricesDiagonal)|isempty(regularizationDiagonal)|isempty(lambdaDiagonal)|isinf(lambdaDiagonal)
    penaltyDiagonal=0;
else
    switch regularizationDiagonal
        case 'Ridge'
            switch matricesDiagonal
                case 'A'
                    penalizationDiagonal=sum((diag(A)-mean(diag(A))).^2);
                case 'B'
                    penalizationDiagonal=sum((diag(B)-mean(diag(B))).^2);
                case 'AB'
                    penalizationDiagonal=sum((diag(A)-mean(diag(A))).^2)+sum((diag(B)-mean(diag(B))).^2);
            end
        case 'LASSO'
            switch matricesDiagonal
                case 'A'
                    penalizationDiagonal=sum(abs(diag(A)-mean(diag(A))));
                case 'B'
                    penalizationDiagonal=sum(abs(diag(B)-mean(diag(B))));
                case 'AB'
                    penalizationDiagonal=sum(abs(diag(A)-mean(diag(A))))+sum(abs(diag(B)-mean(diag(B))));
            end
    end    
    penaltyDiagonal=penalizationDiagonal*lambdaDiagonal;
end

if isempty(matricesFull)|isempty(regularizationFull)|isempty(lambdaFull)|isinf(lambdaFull)
    penaltyFull=0;
else
    switch regularizationFull
        case 'Ridge'
            switch matricesFull
                case 'A'
                    penalizationFull=sum(A(:).^2)-sum(diag(A).^2);
                case 'B'
                    penalizationFull=sum(B(:).^2)-sum(diag(B).^2);
                case 'AB'
                    penalizationFull=sum(A(:).^2)-sum(diag(A).^2)+sum(B(:).^2)-sum(diag(B).^2);
            end
        case 'LASSO'
            switch matricesFull
                case 'A'
                    penalizationFull=sum(abs(A(:)))-sum(abs(diag(A)));
                case 'B'
                    penalizationFull=sum(abs(B(:)))-sum(abs(diag(B)));
                case 'AB'
                    penalizationFull=sum(abs(A(:)))-sum(abs(diag(A)))+sum(abs(B(:)))-sum(abs(diag(B)));
            end
    end
    penaltyFull=penalizationFull*lambdaFull;
end

penalty=penaltyDiagonal+penaltyFull;

end