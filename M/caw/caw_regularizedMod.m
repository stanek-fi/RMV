function [parameters, ll, Ht, output, VCV, scores, DiagnosticsDiagonal, DiagnosticsLDiagonal, DiagnosticsFull, DiagnosticsLFull] = caw_regularized(data,dataAsym,p,o,q,type,startingVals,options,lambdasDiagonal,matricesDiagonal,regularizationDiagonal,lambdasFull,matricesFull,regularizationFull,rT0,lossFunction)

% data=datae
% dataAsym=[]
% % startingVals=startingValsDiagonal
% startingVals=startingValsFull

if ndims(data)==2
    [T,k] = size(data);
    data_rest=data(1:rT0,:);
    data_rf=data((rT0+1):T,:);
else
    [k,~,T] = size(data);
    data_rest=data(:,:,1:rT0);
    data_rf=data(:,:,(rT0+1):T);
end



if type>=2
    lambdasDiagonal=sort(lambdasDiagonal,'descend');
    Nrow=length(lambdasDiagonal);
    DiagnosticsDiagonal=table(lambdasDiagonal',NaN(Nrow,1),NaN(Nrow,1),NaN(Nrow,1),NaN(Nrow,1),NaN(Nrow,1),zeros(Nrow,1),'VariableNames',{'Lambda','OutFit','InFit','Penalty','ll','Iterations','Selected'});
    LDiagonal=NaN(T,Nrow);
    
    [parameters,ll,H,output]=caw(data_rest,dataAsym,p,o,q,1,[],options);
    [C,A,~,B,v]=caw_parameter_transform(parameters,p,o,q,k,1);
    rstartingVals=[parameters(1:((k^2-k)/2+k)); diag(A); diag(B);v];
    H_rf=caw_forecast(data,H,parameters,p,o,q,1);
    
    for i=1:Nrow
        strcat("Diagonal lambda=",num2str(lambdasDiagonal(i)))
        if ~isinf(lambdasDiagonal(i))
            [parameters,ll,H,output]=caw(data_rest,dataAsym,p,o,q,2,rstartingVals,options,DiagnosticsDiagonal.Lambda(i),matricesDiagonal,regularizationDiagonal,0,matricesFull,regularizationFull);
            [C,A,~,B,v]=caw_parameter_transform(parameters,p,o,q,k,2);
%             rstartingVals=[parameters(1:((k^2-k)/2+k)); diag(A); diag(B);v];
            H_rf=caw_forecast(data,H,parameters,p,o,q,2);
        end
        
        DiagnosticsDiagonal.ll(i)=ll;
        [DiagnosticsDiagonal.InFit(i),LDiagonal(1:rT0,i)]=aux_loss(data_rest,H,lossFunction);
        [DiagnosticsDiagonal.OutFit(i),LDiagonal((rT0+1):T,i)]=aux_loss(data_rf,H_rf,lossFunction);
        DiagnosticsDiagonal.Penalty(i)=caw_penalty(C,A,B,1,matricesDiagonal,regularizationDiagonal,0,matricesFull,regularizationFull);
        DiagnosticsDiagonal.Iterations(i)=output.iterations;
    end
    iminDiagonal=find(min(DiagnosticsDiagonal.OutFit)==DiagnosticsDiagonal.OutFit,1,'last');
    DiagnosticsDiagonal.Selected(iminDiagonal)=1;
    lambdaDiagonal=DiagnosticsDiagonal.Lambda(iminDiagonal);

    time=1:T;
    sample=zeros(1,T);
    sample((rT0+1):T)=1;
    DiagnosticsLDiagonal=[array2table(time','VariableNames',cellstr('time')),...
        array2table(sample','VariableNames',cellstr('sample')),...
        array2table(LDiagonal,'VariableNames',cellstr(strcat('lambda_',strsplit(num2str(1:Nrow)))))];
else
    error('for regularized CAW, type must be either 2 (diagonal CAW) or 3 (full CAW)')
end


if type>=3
    lambdasFull=sort(lambdasFull,'descend');
    Nrow=length(lambdasFull);
    DiagnosticsFull=table(lambdasFull',NaN(Nrow,1),NaN(Nrow,1),NaN(Nrow,1),NaN(Nrow,1),NaN(Nrow,1),zeros(Nrow,1),'VariableNames',{'Lambda','OutFit','InFit','Penalty','ll','Iterations','Selected'});
    LFull=NaN(T,Nrow);
    
    [parameters,ll,H,output]=caw(data_rest,dataAsym,p,o,q,2,rstartingVals,options,lambdaDiagonal,matricesDiagonal,regularizationDiagonal,0,matricesFull,regularizationFull);
%     [parameters,ll,H,output]=caw(data_rest,dataAsym,p,o,q,2,rstartingVals,options,0,matricesDiagonal,regularizationDiagonal,0,matricesFull,regularizationFull);
    [C,A,~,B,v]=caw_parameter_transform(parameters,p,o,q,k,2);
    rstartingVals=[parameters(1:((k^2-k)/2+k)); A(:); B(:);v];
    H_rf=caw_forecast(data,H,parameters,p,o,q,2);
    
    for i=1:Nrow
        strcat("Full lambda=",num2str(lambdasFull(i)))
        if ~isinf(lambdasFull(i))
            [parameters,ll,H,output]=caw(data_rest,dataAsym,p,o,q,3,rstartingVals,options,lambdaDiagonal,matricesDiagonal,regularizationDiagonal,DiagnosticsFull.Lambda(i),matricesFull,regularizationFull);
%             [parameters,ll,H,output]=caw(data_rest,dataAsym,p,o,q,3,rstartingVals,options,0,matricesDiagonal,regularizationDiagonal,DiagnosticsFull.Lambda(i),matricesFull,regularizationFull);
            [C,A,~,B,v]=caw_parameter_transform(parameters,p,o,q,k,3);
%             rstartingVals=[parameters(1:((k^2-k)/2+k)); A(:); B(:);v];
            H_rf=caw_forecast(data,H,parameters,p,o,q,3);
        end
        
        DiagnosticsFull.ll(i)=ll;
        [DiagnosticsFull.InFit(i),LFull(1:rT0,i)]=aux_loss(data_rest,H,lossFunction);
        [DiagnosticsFull.OutFit(i),LFull((rT0+1):T,i)]=aux_loss(data_rf,H_rf,lossFunction);
        DiagnosticsFull.Penalty(i)=caw_penalty(C,A,B,0,matricesDiagonal,regularizationDiagonal,1,matricesFull,regularizationFull);
        DiagnosticsFull.Iterations(i)=output.iterations;
    end
    iminFull=find(min(DiagnosticsFull.OutFit)==DiagnosticsFull.OutFit,1,'last');
    DiagnosticsFull.Selected(iminFull)=1;
    lambdaFull=DiagnosticsFull.Lambda(iminFull);
    
    time=1:T;
    sample=zeros(1,T);
    sample((rT0+1):T)=1;
    DiagnosticsLFull=[array2table(time','VariableNames',cellstr('time')),...
        array2table(sample','VariableNames',cellstr('sample')),...
        array2table(LFull,'VariableNames',cellstr(strcat('lambda_',strsplit(num2str(1:Nrow)))))];
else
    lambdaFull=0;
    DiagnosticsFull=[];
    DiagnosticsLFull=[];
end

strcat("Final estimation: lambdaDiagonal=",num2str(lambdaDiagonal)," lambdaFull=", num2str(lambdaFull))
[parameters, ll, Ht, output, VCV, scores] = caw(data,dataAsym,p,o,q,type,startingVals,options,lambdaDiagonal,matricesDiagonal,regularizationDiagonal,lambdaFull,matricesFull,regularizationFull);

end

