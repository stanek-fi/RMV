function [parameters, ll, Ht, output, VCV, scores] = bekk(data,dataAsym,p,o,q,type,startingVals,options,lambdaDiagonal,matricesDiagonal,regularizationDiagonal,lambdaFull,matricesFull,regularizationFull)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Argument Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<1
    error('3 to 8 inputs required.')
end
[T,k] = size(data);
if ndims(data)==3
    [k,~,T] = size(data);
end

switch nargin
    case 5
        type = 1;
        startingVals = [];
        options = [];
        lambdaDiagonal=[];
        matricesDiagonal=[];
        regularizationDiagonal=[];
        lambdaFull=[];
        matricesFull=[];
        regularizationFull=[];    
    case 6
        startingVals = [];
        options = [];
        lambdaDiagonal=[];
        matricesDiagonal=[];
        regularizationDiagonal=[];
        lambdaFull=[];
        matricesFull=[];
        regularizationFull=[];    
    case 7
        options = [];
        lambdaDiagonal=[];
        matricesDiagonal=[];
        regularizationDiagonal=[];
        lambdaFull=[];
        matricesFull=[];
        regularizationFull=[];    
    case 8
        lambdaDiagonal=[];
        matricesDiagonal=[];
        regularizationDiagonal=[];
        lambdaFull=[];
        matricesFull=[];
        regularizationFull=[];       
    case 14
        
    otherwise
        error('5 to 8 (or 14 in the case of regularization) inputs required ')
end

if ndims(data)>3
    error('DATA must be either a T by K matrix or a K by K by T array.')
end
if T<=k
    error('DATA must be either a T by K matrix or a K by K by T array, and T must be larger than K.')
end
if ndims(data)==3 && o>0
    if isempty(dataAsym) || ndims(dataAsym)~=3 || any(size(dataAsym)~=size(data))
        error('DATAASYM must be provided when O>0 and DATA is a 3D array.')
    end
end

if p<1 || floor(p)~=p
    error('P must be a positive scalar.')
end
if isempty(o)
    o=0;
end
if o<0 || floor(o)~=o
    error('O must be a non-negative scalar.')
end
if isempty(q)
    q=0;
end
if q<0 || floor(q)~=q
    error('Q must be a non-negative scalar.')
end

% if strcmpi(type,'Scalar')
%     type = 1;
% elseif strcmpi(type,'Diagonal')
%     type = 2;
% elseif strcmpi(type,'Full')
%     type = 3;
% else
%     error('TYPE must be ''Scalar'', ''Diagonal'' or ''Full''.')
% end

k2 = k*(k+1)/2;
if ~isempty(startingVals)
    switch type
        case 1
            count = p+o+q;
        case 2
            count = (p+o+q)*k;
        case 3
            count = (p+o+q)*k*k;
    end
    count = count + k2;
    if length(startingVals)~=count
        error('STARTINGVALS does not have the expected number of elements.')
    end
end

if isempty(options)
    options = optimset('fmincon');
    options.Display = 'iter';
    options.Diagnostics = 'on';
    options.Algorithm = 'interior-point';
else
    try
        options = optimset(options);
    catch ME
        error('The user provided options structure is not valid.')
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data Transformation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ndims(data)==2
    temp = zeros(k,k,T);
    for i=1:T
        temp(:,:,i) = data(i,:)'*data(i,:);
        eta = (data(i,:).*(data(i,:)<0));
        dataAsym(:,:,i) =  eta'*eta;
    end
    data = temp;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Starting Values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(startingVals)
    startingOptions = optimset('fminunc');
    startingOptions.LargeScale = 'off';
    startingOptions.Display = 'none';
    [startingVals,~,~,intercept] = scalar_vt_vech(data,dataAsym,p,o,q,[],[],startingOptions);
    C = intercept;
    C = chol2vec(chol(C)');
    switch type
        case 1
            shape = 1;
        case 2
            shape = ones(k,1);
        case 3
            shape = eye(k);
    end
    sv = [];
    for i=1:p+o+q
        temp = sqrt(startingVals(i));
        temp = temp * shape;
        sv = [sv;temp(:)]; %#ok<AGROW>
    end
    startingVals = [C;sv];
end
UB = .99998 * ones(size(startingVals));
UB(1:k2) = inf;
LB = -UB;
m = ceil(sqrt(T));
w = .06 * .94.^(0:(m-1));
w = reshape(w/sum(w),[1 1 m]);
backCast = sum(bsxfun(@times,w,data(:,:,1:m)),3);
if o>0
    backCastAsym = sum(bsxfun(@times,w,dataAsym(:,:,1:m)),3);
else
    backCastAsym = [];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use stdData and set C = eye(K)
%rarch_likelihood(startingVals,data,p,q,C,backCast,type,false,false);


if type==1
    Aeq=[];
    beq=[];
end

if type==2
    if isinf(lambdaDiagonal)
        Restricted=eye(k);
        Restricted=[zeros(((k^2-k)/2+k),1); diag(Restricted); diag(Restricted)];
        indexRestricted=find(Restricted);
        AeqDiagonal=zeros(length(indexRestricted)-2,length(Restricted));
        for b=1:2
            for r=1:(k-1)
                AeqDiagonal((b-1)*(k-1)+r,indexRestricted((b-1)*k+1))=1;
                AeqDiagonal((b-1)*(k-1)+r,indexRestricted((b-1)*k+1+r))=-1;
            end
        end
        beqDiagonal=zeros(length(indexRestricted)-2,1);
    else
        AeqDiagonal=[];
        beqDiagonal=[];
    end
    
    if isinf(lambdaFull)
        AeqFull=[];
        beqFull=[];
    else
        AeqFull=[];
        beqFull=[];
    end

    Aeq=[AeqDiagonal;AeqFull];
    beq=[beqDiagonal;beqFull];
end

if type==3
    if isinf(lambdaDiagonal)
        Restricted=eye(k);
        Restricted=[zeros(((k^2-k)/2+k),1); Restricted(:); Restricted(:)];
        indexRestricted=find(Restricted);
        AeqDiagonal=zeros(length(indexRestricted)-2,length(Restricted));
        for b=1:2
            for r=1:(k-1)
                AeqDiagonal((b-1)*(k-1)+r,indexRestricted((b-1)*k+1))=1;
                AeqDiagonal((b-1)*(k-1)+r,indexRestricted((b-1)*k+1+r))=-1;
            end
        end
        beqDiagonal=zeros(length(indexRestricted)-2,1);
    else
        AeqDiagonal=[];
        beqDiagonal=[];
    end
    
    if isinf(lambdaFull)
        Restricted=1-eye(k);
        Restricted=[zeros(((k^2-k)/2+k),1); Restricted(:); Restricted(:)];
        indexRestricted=find(Restricted);
        AeqFull=zeros(length(indexRestricted),length(Restricted));
        for r=1:length(indexRestricted)
            AeqFull(r,indexRestricted(r))=1;
        end
        beqFull=zeros(length(indexRestricted),1);
    else
        AeqFull=[];
        beqFull=[];
    end
    
    Aeq=[AeqDiagonal;AeqFull];
    beq=[beqDiagonal;beqFull];
end


warning('off') %#ok<*WNOFF>
[parameters,ll,~,output] = fmincon(@bekk_likelihood,startingVals,[],[],Aeq,beq,[],[],[],options,data,dataAsym,p,o,q,backCast,backCastAsym,type,lambdaDiagonal,matricesDiagonal,regularizationDiagonal,lambdaFull,matricesFull,regularizationFull);
% [parameters,ll,~,output] = fminunc(@caw_likelihood,startingVals,options,data,dataAsym,p,o,q,backCast,backCastAsym,type,lambdaDiagonal,matricesDiagonal,regularizationDiagonal,lambdaFull,matricesFull,regularizationFull);
warning('on') %#ok<*WNON>

[ll,~,Ht] = bekk_likelihood(parameters,data,dataAsym,p,o,q,backCast,backCastAsym,type,0,[],[],0,[],[]);
ll = -ll;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inference
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargout>=5
    [VCV,~,~,scores] = robustvcv(@bekk_likelihood,parameters,0,data,dataAsym,p,o,q,backCast,backCastAsym,type,lambdaDiagonal,matricesDiagonal,regularizationDiagonal,lambdaFull,matricesFull,regularizationFull);
end
