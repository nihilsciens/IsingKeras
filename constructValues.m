%% Initialization

% Parametrar
J  = 1;
numSpinsPerDim = 10;
probSpinUp = 0.5;
spin = sign(probSpinUp - rand(numSpinsPerDim, numSpinsPerDim));

% Första värde
kT = 0;

% Utskriftsparametar
formatSpec1 = '%d,';
formatSpec2 = '%d\n';
fileID = fopen('IsingValues.txt','w+');

%%%%%%%%%%%%%%
% Algoritmen %
%%%%%%%%%%%%%%
numIters = 10^7 * numel(spin) * 2;
for iter = 1 : numIters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Fysikaliska operationer %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    linearIndex = randi(numel(spin));
    [row, col]  = ind2sub(size(spin), linearIndex);
    above = mod(row - 1 - 1, size(spin,1)) + 1;
    below = mod(row + 1 - 1, size(spin,1)) + 1;
    left  = mod(col - 1 - 1, size(spin,2)) + 1;
    right = mod(col + 1 - 1, size(spin,2)) + 1;
    neighbors = [spin(above,col);spin(row,left);spin(row,right);spin(below,col)];
    dE = 2 * J * spin(row, col) * sum(neighbors);
    prob = exp(-dE / kT);
    if dE <= 0 || rand() <= prob
        spin(row, col) = - spin(row, col);
    end
    
    %%%%%%%%%%%%%%%%%%
    % Skriv till fil %
    %%%%%%%%%%%%%%%%%%
    if(mod(iter, 10^5) == 0)
        if kT == 0
            fprintf(fileID,formatSpec1,spin);
            fprintf(fileID,'%d,%d\n',[1 0]);
            kT = 100;
        else
            fprintf(fileID,formatSpec1,spin);
            fprintf(fileID,'%d,%d\n',[0 1]);
            kT = 0;
        end
        iter/(10^5)
        if iter/(10^5) == 200
            break;
        end
    end
end
