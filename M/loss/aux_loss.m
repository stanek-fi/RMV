function [l,L,Hr] = aux_loss(data,Hf,lossFunction)

[k,~,T]=size(Hf);
if ndims(data)==2
    Hr=NaN(k,k,T);
    for t=1:T
        Hr(:,:,t)=(data(t,:)'*data(t,:));
    end
else
    Hr=data;
end


L=NaN(T,1);
switch lossFunction
    case 'Euclidean'
        for t=1:T
            Hfsimetric=(Hf(:,:,t)+Hf(:,:,t)')/2;
            L(t)=vech(Hr(:,:,t)-Hfsimetric)'*vech(Hr(:,:,t)-Hfsimetric);
        end
    case 'Frobenius'
        for t=1:T
            L(t)=trace((Hr(:,:,t)-Hf(:,:,t))'*(Hr(:,:,t)-Hf(:,:,t)));
        end
    case 'Stein'
        for t=1:T
            L(t)=trace(Hf(:,:,t)\Hr(:,:,t))-log(abs(det(Hf(:,:,t)\Hr(:,:,t))))-k;
        end
    case 'minusBEKKll'
        logLikConst = k*log(2*pi);
        for t=1:T
            L(t)=0.5*(logLikConst + log(det(Hf(:,:,t))) + sum(diag(Hf(:,:,t)^(-1)*Hr(:,:,t))));
        end
    case 'minusCAWll'
        %v=12.1075;
        v=27.2492;
        logLikConst=(v*k/2*log(2)+k*(k-1)/4*log(pi));
        for ik=1:k
            logLikConst=logLikConst+log(gamma((v+1-ik)/2));
        end
        for t=1:T
            L(t) = logLikConst + 0.5*(v*log(det(Hf(:,:,t)/v)) - (v-k-1)*log(det(Hr(:,:,t))) + sum(diag(v*Hf(:,:,t)^(-1)*Hr(:,:,t))));
        end
end
l=mean(L);
end
