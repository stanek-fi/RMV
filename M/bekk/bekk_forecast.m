function [Hf] = bekk_forecast(data,H,parameters,p,o,q,type)

if ndims(data)==2
    [T,k] = size(data);
    temp = zeros(k,k,T);
    for i=1:T
        temp(:,:,i) = data(i,:)'*data(i,:);
    end
    data = temp;
else
    [k,~,T] = size(data);
end
[~,~,T0]=size(H);
[C,A,~,B]=bekk_parameter_transform(parameters,p,o,q,k,type);

Ht=cat(3,H,NaN(k,k,T-T0));
for i=(T0+1):T
    Ht(:,:,i) = C;
    for j=1:p
            Ht(:,:,i) = Ht(:,:,i) + A(:,:,j)'*data(:,:,i-j)*A(:,:,j);
    end
    for j=1:q
            Ht(:,:,i) = Ht(:,:,i) + B(:,:,j)'*Ht(:,:,i-j)*B(:,:,j);
    end
end

Hf=Ht(:,:,(T0+1):T);

end