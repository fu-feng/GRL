import pickle

def save_genepool(genepool, average_reward, generation_id, args):
    save_path = (args.out_dir + args.generation_dir + '/genepool{}.txt').format(generation_id, generation_id)
    file = open(save_path, 'w')
    file.write("Generation %d\t Average reward: %.2f\n"%(generation_id,average_reward))
    for key, value in genepool.logits.items():
        pool = "{}\t{:.3f}\t".format(key, value)
        for a in genepool.genepool[key]:
            pool += '{}_{}_{}: {:.2f}\t\t'.format(a.generation_id, a.agent_id, a.task_id, a.score)
        pool += '\n'
        file.write(pool)
    file.close()

def save_agent_rewards(score_dict, generation_id, agent_ids, gene, args):
    score_list = []
    save_path = (args.out_dir + args.generation_dir + '/score_of_gen{}.txt').format(generation_id, generation_id)
    file = open(save_path, 'w')
    for key, value in score_dict.items():
        score = (key, value)
        score_list.append(score)
    score_list.sort(key=lambda x:int(x[0].split('/')[-1][:-4].split('_')[1]))
    for i in range(len(score_list)):
        if gene:
            file.write(score_list[i][0].split('/')[-1][:-4] + '\t{:.1f}\t'.format(score_list[i][1]) + str(gene[int(score_list[i][0].split('/')[-1].split('_')[1])][0]) + '\t' + str(gene[int(score_list[i][0].split('/')[-1].split('_')[1])][1].split('/')[-1][:-4]) + '\n')
        else:
            file.write(score_list[i][0].split('/')[-1][:-4] + '\t{:.1f}\t'.format(score_list[i][1]) + '\n')
    return score_list

def save_checkpoint(genepool, i, best_reward, patience, args):
    checkpoint = {
            "genepool":genepool,
            "i":i,
            "best_reward":best_reward,
            "patience":patience,
        }
    with open(args.checkpoint_dir.format(i, i), "wb") as f:
        pickle.dump(checkpoint,f)
