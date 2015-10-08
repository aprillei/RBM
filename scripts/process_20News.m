function process_20News(num_features, num_training_examples, num_test_examples)
    % Process the downloaded dataset to keep num_features, 
    % num_training_examples, num_test_examples

    % Load the dataset
    load('../dataset/20Newsgroups.mat')
    
    % Make the features matrix full
    fea = full(fea);
    
    % Truncate the features matrix to keep num_features only
    trunc_fea = fea(:,1:num_features);
    
    % Separate train and test sets
    ft_train = trunc_fea(trainIdx,:);
    ft_test = trunc_fea(testIdx,:);
    
    gnd_train = gnd(trainIdx,:);
    gnd_test = gnd(testIdx,:);
    
    % Transform into binary features (training a binary RBM)
    ft_train_bin = double((ft_train ~= 0));
    ft_test_bin = double((ft_test ~= 0));
    
    % Write the first num_training_examples features to file
    fid = fopen('../dataset/train.fea', 'w+');
    for i=1:num_training_examples
       fprintf(fid, '%d ', ft_train_bin(i,:));
       fprintf(fid, '\n');
    end
    fclose(fid);
    % Write the first num_training_examples labels to file
    fid = fopen('../dataset/train.labels', 'w+');
    for i=1:num_training_examples
       fprintf(fid, '%d ', gnd_train(i,:));
       fprintf(fid, '\n');
    end
    fclose(fid);
    
    % Write the first num_test_examples features to file
    fid = fopen('../dataset/test.fea', 'w+');
    for i=1:num_test_examples
       fprintf(fid, '%d ', ft_test_bin(i,:));
       fprintf(fid, '\n');
    end
    fclose(fid);
    
    % Write the first num_test_examples labels to file
    fid = fopen('../dataset/test.labels', 'w+');
    for i=1:num_test_examples
       fprintf(fid, '%d ', gnd_test(i,:));
       fprintf(fid, '\n');
    end
    fclose(fid);
end
