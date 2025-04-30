from cmgrdf_cli import cpp

def declare(file=None, sig=None, bkg=None):
    cpp_declare = """
        #ifndef PT_HIST_H__
        #define PT_HIST_H__
        TFile *f = TFile::Open("<file>");

        //check if <sig> is in the file
        if (!f->GetListOfKeys()->Contains("<sig>")) {
            std::cout << "The file does not contain the <sig> histogram" << std::endl;
            return;
        }
        if (!f->GetListOfKeys()->Contains("<bkg>")) {
            std::cout << "The file does not contain the <bkg> histogram" << std::endl;
            return;
        }

        TH1F *ratio_h = (TH1F*)f->Get("<sig>");
        TH1F *bkg_h = (TH1F*)f->Get("<bkg>");
        ratio_h->Divide(bkg_h);


        ROOT::RVec<float> reweight_pt_h(ROOT::RVec<float> pt) {
            ROOT::RVec<float> weight(pt.size());
            for (int i = 0; i < pt.size(); i++) {
                int bin = ratio_h->FindBin(pt[i]);
                weight[i]=ratio_h->GetBinContent(bin);
            }
            return weight;
        }
        #endif

    """.replace("<file>", file).replace("<sig>", sig).replace("<bkg>", bkg)
    cpp.declare(cpp_declare)


