如何產出Source資料
請進入Data資料夾
HIT_cilin_utf8CT_zhconv為原始同義詞行檔案
使用delRepeat將其重複刪除。成為FIX_HIT_cilin_utf8CT_zhconv檔案


====================================若已有Source資料
請先執行 divide_dataset.py檔案，產出的檔案位於 divide_dataset_output資料夾中 此為生產train、test與valid之同義詞行

之後運行產出pairs的程式，分別為smart_cross與generate_pairs_data(皆位於大資料夾內)，若使用generate_pairs_data，必須再從pairs_output資料夾中運行split_to_datasets，爾後在model_datasets就是最終資料目的地。 而smart_cross輸出位於smart_cross_output。
