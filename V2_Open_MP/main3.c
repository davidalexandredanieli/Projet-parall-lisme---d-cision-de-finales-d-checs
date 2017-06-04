#include "projet.h"
#include <sys/time.h>
#include <omp.h>
/* 2017-02-23 : version 2.0 OpenMP */

double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

unsigned long long int node_searched = 0;

void evaluate(tree_t * T, result_t *result, int* continuer)
{

        node_searched++;
  
        move_t moves[MAX_MOVES];
        int n_moves;

        result->score = -MAX_SCORE - 1;
        result->pv_length = 0;
        
        if (test_draw_or_victory(T, result))
          return;

        if (TRANSPOSITION_TABLE && tt_lookup(T, result))     /* la réponse est-elle déjà connue ? */
          return;
        
        compute_attack_squares(T);

        /* profondeur max atteinte ? si oui, évaluation heuristique */
        if (T->depth == 0) {
          result->score = (2 * T->side - 1) * heuristic_evaluation(T);
          return;
        }
        
        n_moves = generate_legal_moves(T, &moves[0]);

        /* absence de coups légaux : pat ou mat */
	if (n_moves == 0) {
          result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
          return;
        }
        
        if (ALPHA_BETA_PRUNING)
          sort_moves(T, n_moves, moves);

        /* évalue récursivement les positions accessibles à partir d'ici */
		if(T->height==0){
					//printf("dans le if(T->height==0)\n");
			#pragma omp parallel for schedule(runtime) shared(continuer)
		    for (int i = 0; i < n_moves; i++) {
			if(*continuer!=0){
				tree_t child;
				        result_t child_result;
				        
				        play_move(T, moves[i], &child);
				        
				        evaluate(&child, &child_result,continuer);
				                 
				        int child_score = -child_result.score;

				if (child_score > result->score) {
					result->score = child_score;
					result->best_move = moves[i];
				                result->pv_length = child_result.pv_length + 1;
				                for(int j = 0; j < child_result.pv_length; j++)
				                  result->PV[j+1] = child_result.PV[j];
				                  result->PV[0] = moves[i];
				        }

				        if (ALPHA_BETA_PRUNING && child_score >= T->beta)
				          	*continuer=0;    
						if(*continuer!=0)
				        	T->alpha = MAX(T->alpha, child_score);
				}
		    }

		}else{
					    for (int i = 0; i < n_moves; i++) {
			tree_t child;
		            result_t child_result;
		            
		            play_move(T, moves[i], &child);
		            
		            evaluate(&child, &child_result,continuer);
		                     
		            int child_score = -child_result.score;

			if (child_score > result->score) {
				result->score = child_score;
				result->best_move = moves[i];
		                    result->pv_length = child_result.pv_length + 1;
		                    for(int j = 0; j < child_result.pv_length; j++)
		                      result->PV[j+1] = child_result.PV[j];
		                      result->PV[0] = moves[i];
		            }

		            if (ALPHA_BETA_PRUNING && child_score >= T->beta)
		              return;    

		            T->alpha = MAX(T->alpha, child_score);
		    }

		}
}


void decide(tree_t * T, result_t *result)
{
	for (int depth = 1;; depth++) {
		T->depth = depth;
		T->height = 0;
		T->alpha_start = T->alpha = -MAX_SCORE - 1;
		int* continuer=(int*)malloc(sizeof(int));
		*continuer=1;
		T->beta = MAX_SCORE + 1;

                printf("=====================================\n");
		//printf("juste avant le evaluate\n");
		evaluate(T, result,continuer);

                printf("depth: %d / score: %.2f / best_move : ", T->depth, 0.01 * result->score);
                print_pv(T, result);
                
                if (DEFINITIVE(result->score))
                  break;
	}
}

int main(int argc, char **argv)
{  
	printf("début\n");
	double debut=0.0, fin=0.0;
	tree_t root;
        result_t result;

        if (argc < 2) {
          printf("usage: %s \"4k//4K/4P w\" (or any position in FEN)\n", argv[0]);
          exit(1);
        }

        if (ALPHA_BETA_PRUNING)
          printf("Alpha-beta pruning ENABLED\n");

        if (TRANSPOSITION_TABLE) {
          printf("Transposition table ENABLED\n");
          init_tt();
        }
        
        parse_FEN(argv[1], &root);
        print_position(&root);
    debut = my_gettimeofday();    
	decide(&root, &result);
	fin = my_gettimeofday();
  	fprintf( stdout, "total computation time (with gettimeofday()) : %g s\n",
	   fin - debut);
	printf("\nDécision de la position: ");
        switch(result.score * (2*root.side - 1)) {
        case MAX_SCORE: printf("blanc gagne\n"); break;
        case CERTAIN_DRAW: printf("partie nulle\n"); break;
        case -MAX_SCORE: printf("noir gagne\n"); break;
        default: printf("BUG\n");
        }
	
        printf("Node searched: %llu\n", node_searched);
        
        if (TRANSPOSITION_TABLE)
          free_tt();
	return 0;
}
