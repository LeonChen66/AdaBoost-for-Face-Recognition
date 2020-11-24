from face_rec.adaboost import AdaBoost
from face_rec.utils import *
from plot_metric.functions import BinaryClassification
from sklearn.metrics import f1_score

if __name__ == "__main__":
    pos_training_path = 'data/trainset/faces'
    neg_training_path = 'data/trainset/non-faces'
    pos_testing_path = 'data/testset/faces'
    neg_testing_path = 'data/testset/non-faces'

    
    # For performance reasons restricting feature size
    min_feature_height =8
    max_feature_height = 10
    min_feature_width = 8
    max_feature_width = 10

    # Load the data
    print('Loading faces..')
    faces_training = load_images(pos_training_path)
    faces_ii_training = list(map(to_integral_image, faces_training))
    print('..done. ' + str(len(faces_training)) + ' faces loaded.\n\nLoading non faces..')
    non_faces_training = load_images(neg_training_path)
    non_faces_ii_training = list(map(to_integral_image, non_faces_training))
    print('..done. ' + str(len(non_faces_training)) + ' non faces loaded.\n')
    print('Loading test faces..')
    faces_testing = load_images(pos_testing_path)
    faces_ii_testing = list(map(to_integral_image, faces_testing))
    print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces_testing = load_images(neg_testing_path)
    non_faces_ii_testing = list(map(to_integral_image, non_faces_testing))
    print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

    for num in ([1, 3, 5, 10, 20]):
        num_classifiers = num
        # classifiers are haar like features
        ab = AdaBoost()
        classifiers = ab.learn(faces_ii_training, non_faces_ii_training, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width)

        print('Testing selected classifiers..')
        correct_faces = 0
        correct_non_faces = 0
        correct_faces = sum(ensemble_vote_all(faces_ii_testing, classifiers))
        incorrect_faces = len(faces_testing) - correct_faces
        correct_non_faces = len(non_faces_testing) - sum(ensemble_vote_all(non_faces_ii_testing, classifiers))
        incorrect_non_faces = len(non_faces_testing) - correct_non_faces
        
        correct_faces_score = ensemble_score_all(faces_ii_testing, classifiers)
        incorrect_non_faces_score = ensemble_score_all(non_faces_ii_testing, classifiers)
        face_label = np.array([1] * len(correct_faces_score) + [0] * len(incorrect_non_faces_score))
        face_predict = np.array(ensemble_vote_all(faces_ii_testing, classifiers) + ensemble_vote_all(non_faces_ii_testing, classifiers))
        
        # plot confusion matrix
        cf = plot_confusion_matrix(correct_faces, incorrect_faces, correct_non_faces, incorrect_non_faces)
        cf.figure.savefig("results/cf_round_{}.png".format(num))
        # plot roc curve
        # Visualisation with plot_metric
        bc = BinaryClassification(face_label, face_predict, labels=["Class 1", "Class 2"])
        f1 = f1_score(face_label, face_predict)
        print("f1: ", f1)
        # Figures
        plt.figure(figsize=(5,5))
        fpr, tpr, thres, auc = bc.plot_roc_curve()
        plt.savefig('results/round_{}_roc.jpg'.format(num))
        print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
            + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
            + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
            + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')
        
        print('False Positive Rate: {}, True Positive Rate: {}'.format(fpr, tpr))
        print('Classifier Results: ', ab.stats)

        haar_imgs = vis_haar(classifiers, faces_testing[0])

        for i, img in enumerate(haar_imgs):
            cv2.imwrite( 'results/top_features/%s_round.png' %i, img)
